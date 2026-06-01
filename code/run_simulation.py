"""
PyTorch benchmark runner for the Drosophila brain model.

Implements the LIF neuron model with alpha-function synapses using PyTorch,
with support for both CPU and CUDA GPU computation. Batches n_run trials
in parallel for efficient GPU utilization.

Model architecture (from Shiu et al.):
    PoissonSpikeGenerator → recurrent weights (sparse matmul) → AlphaLIF
    where AlphaLIF = AlphaSynapse + LIFNeuron + refractory period

Called by benchmark.py orchestrator.
"""

import pandas as pd
import pyarrow  # noqa: F401  — must be imported before torch to avoid libarrow conflict
import numpy as np
import torch
import time
import torch.nn as nn
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import (
    ACTIVE_NEURON_DOT_SIZE,
    ACTIVE_NEURON_ALPHA,
    HIGHLIGHT_NEURON_DOT_SIZE,
    plot_brain_ghost,
)

# ============================================================================
# PyTorch Model Parameters (matching Brian2 default_params)
# ============================================================================

MODEL_PARAMS = {
    "tauSyn": 5.0,  # ms
    "tDelay": 1.8,  # ms
    "v0": -52.0,  # mV
    "vReset": -52.0,  # mV
    "vRest": -52.0,  # mV
    "vThreshold": -45.0,  # mV
    "tauMem": 20.0,  # ms
    "tRefrac": 2.2,  # ms
    "scalePoisson": 250,
    "wScale": 0.275,
    # TODO: Fiddle with these values.
    # How much the history should decay after a second (multiplier).
    "ltpHistoryDecay": 0.1,
    # How much the LTP connections should decay after a second (multiplier).
    "ltpDecay": 0.1,
    # The maximum amount LTP can multiply a weight by.
    # The true max is always one plus this value, since it's
    # calculated as weights + ltp * weights.
    "ltpMax": 1.0,
    # How much to multiply changes in conductance by to get the
    # ltp multiplier (e.g., 1 / (# of spikes * # of connections * wScale) that
    # should cause the synapse strangth to instantly double).
    "ltpConductanceMul": 1.0 / (10 * 0.275),
}

DT = 0.1  # Simulation timestep in ms (matches Brian2 defaultclock.dt)
N_TRAINING_TRIALS = 10
LTD_RATE = 0.05
DOPAMINE_TYPE = "reward"  # Toggle betwen reward and punishment
AVOIDANCE_MBON_TYPES = {"MBON01", "MBON03", "MBON04", "MBON05"}
APPROACH_MBON_TYPES = {"MBON08", "MBON09", "MBON11", "MBON12", "MBON15", "MBON16", "MBON17", "MBON18","MBON19"}
ODOR_ORN_TYPE = "ORN_DC3"
DOOR_RESPONSE_VALUE = 0.078
ORN_RATE_SCALE = 100000.0   # Tune this if no KCs spike or too many neurons spike.
ODOR_ORN_RATE = DOOR_RESPONSE_VALUE * ORN_RATE_SCALE

SEED = 123

# np.random.seed(SEED)
# torch.manual_seed(SEED)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Model Classes
# ============================================================================


class PoissonSpikeGenerator(nn.Module):
    """Generates one timestep of Poisson-distributed spikes from firing rates."""

    def __init__(self, dt, scale, device="cpu"):
        super().__init__()
        self.prob_scale = dt / 1000.0
        self.scale = scale
        self.device = device

    def forward(self, rates, generator=None):
        return (
            torch.bernoulli(rates * self.prob_scale, generator=generator) * self.scale
        )


class AlphaSynapse(nn.Module):
    """Alpha-function synapse dynamics with configurable delay."""

    def __init__(self, batch, size, dt, params, device="cpu"):
        super().__init__()
        self.time_factor = dt / params["tauSyn"]
        self.steps_delay = int(params["tDelay"] / dt)
        self.size = size
        self.device = device
        self.batch = batch

    def state_init(self):
        conductance = torch.zeros(self.batch, self.size, device=self.device)
        delay_buffer = torch.zeros(
            self.batch, self.steps_delay + 1, self.size, device=self.device
        )
        return conductance, delay_buffer

    def forward(self, input_, conductance, delay_buffer, refrac):
        conductance_new = (
            conductance * (1 - self.time_factor) + delay_buffer[:, 0, :] * refrac
        )
        delay_buffer = torch.roll(delay_buffer, shifts=-1, dims=1)
        delay_buffer[:, -1, :] = input_
        return conductance_new, delay_buffer


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient (ATan)."""

    def __init__(self, batch, size, dt, params, device="cpu"):
        super().__init__()
        self.size = size
        self.dt = dt
        self.tau_mem = params["tauMem"]
        self.v_reset = params["vReset"]
        self.v_rest = params["vRest"]
        self.v_threshold = params["vThreshold"]
        self.v_0 = params["v0"]
        self.time_factor = dt / self.tau_mem
        self.spike_gradient = self.ATan.apply
        self.device = device
        self.batch = batch

    def state_init(self):
        v = torch.zeros(self.batch, self.size, device=self.device) + self.v_0
        spikes = torch.zeros(self.batch, self.size, device=self.device)
        return spikes, v

    def forward(self, input_current, v):
        v = v + self.time_factor * (input_current - (v - self.v_rest))
        spike = self.spike_gradient(v - self.v_threshold)
        reset = ((v - self.v_reset) * spike).detach()
        v = v - reset
        return spike, v

    @staticmethod
    class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v):
            spike = (v > 0).float()
            ctx.save_for_backward(v)
            return spike

        @staticmethod
        def backward(ctx, grad_output):
            (v,) = ctx.saved_tensors
            grad = 1 / (1 + (np.pi * v).pow_(2)) * grad_output
            return grad


class AlphaLIF(nn.Module):
    """LIF neuron with alpha-function synapse dynamics and refractory period."""

    def __init__(self, batch, size, dt, params, device="cpu"):
        super().__init__()
        self.size = size
        self.synapse = AlphaSynapse(batch, size, dt, params, device=device)
        self.neuron = LIFNeuron(batch, size, dt, params, device=device)
        self.steps_refrac = int(params["tRefrac"] / dt)

    def state_init(self):
        conductance, delay_buffer = self.synapse.state_init()
        spikes, v = self.neuron.state_init()
        refrac = self.steps_refrac + torch.zeros_like(v)
        return conductance, delay_buffer, spikes, v, refrac

    def forward(self, input_, conductance, delay_buffer, spikes, v, refrac):
        refrac = refrac * (1 - spikes)
        refrac = refrac + 1
        conductance_new, delay_buffer = self.synapse(
            input_, conductance, delay_buffer, (refrac > self.steps_refrac).float()
        )
        spikes, v_new = self.neuron(conductance, v)
        conductance_reset = (conductance_new * spikes).detach()
        conductance_new = conductance_new - conductance_reset
        return conductance_new, delay_buffer, spikes, v_new, refrac


class TorchModel(nn.Module):
    """
    Top-level model: Poisson input + recurrent connectome weights + AlphaLIF.

    The weights tensor should be a sparse matrix (CSR or COO) derived from
    the Drosophila connectome.
    """

    def __init__(
        self, batch, size, dt, params, weights, enable_ltp: bool, device="cpu"
    ):
        super().__init__()
        self.neurons = AlphaLIF(batch, size, dt, params, device=device)
        self.weights = weights.coalesce()
        # A sum of recent spiking activity, such that each
        # item in the matrix contains the # of spikes from
        # the presynaptic neuron * the weight of the connection,
        # weighted so that older spikes are smaller.
        self.w_history = torch.zeros_like(self.weights).coalesce()
        # A weight multiplier for each connection, that's increased
        # when the connection causes the neuron to spike and decays
        # over time.
        self.ltp_mul = torch.zeros_like(self.weights).coalesce()
        self.poisson = PoissonSpikeGenerator(dt, params["scalePoisson"], device=device)
        self.scale = params["wScale"]
        # History decay is for one second, so we need to adjust it to
        # work for the timestep size.
        self.history_decay = params["ltpHistoryDecay"] ** (DT / 1000)
        self.ltp_decay = params["ltpDecay"] ** (DT / 1000)
        self.ltp_max = params["ltpMax"]
        self.ltp_conductance_mul = params["ltpConductanceMul"]
        self.enable_ltp = enable_ltp

    def state_init(self):
        return self.neurons.state_init()

    def forward(
        self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None
    ):
        if self.enable_ltp:
            self.w_history = self.w_history * self.history_decay
            self.ltp_mul = self.ltp_mul * self.ltp_decay

            # We need to count how much input each neuron is receiving
            # and from which neurons they're receiving it from.
            # That way, when a neuron does spike, we can "credit" the ones
            # that sent the most signals and increase their connections.
            indices = self.weights.indices()
            values = self.weights.values()

            # indicies[1] is a list of all the column indices, so this
            # multiplies the vector with each row, row-by-row.
            #
            # The original code is implemented with multiple simultaneous
            # runs in mind, and so spike's first axis is the run index.
            # For simplicity, I'm not implementing that for LTP, so this
            # will only use the first run and will fail if used with multiple.
            values *= spikes[0][indices[1]] * self.scale

            add_matrix = torch.sparse_coo_tensor(
                indices, values, self.weights.shape, is_coalesced=True
            )
            self.w_history += add_matrix

            self.w_history = self.w_history.coalesce()

        spikes_input = self.poisson(rates, generator=generator)

        if self.enable_ltp:
            new_weights = self.weights + self.weights * self.ltp_mul
            weighted_spikes = torch.sparse.mm(new_weights, spikes.T).T
        else:
            weighted_spikes = torch.sparse.mm(self.weights, spikes.T).T

        conductance, delay_buffer, new_spikes, v, refrac = self.neurons(
            self.scale * (spikes_input + weighted_spikes),
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac,
        )

        if self.enable_ltp:
            # We need to strengthen the connections between neurons
            # that just spiked and the neurons that caused them to spike.
            indices = self.w_history.indices()
            values = self.w_history.values()

            # We only want to strengthen the rows (connections to
            # the neuron that just spiked), which indices[0] corresponds
            # to.
            #
            # See the comment above for why this needs to be new_spikes[0].
            values *= new_spikes[0][indices[0]] * self.ltp_conductance_mul

            add_matrix = torch.sparse_coo_tensor(
                indices, values, self.weights.shape, is_coalesced=True
            )
            self.ltp_mul += add_matrix
            self.ltp_mul = self.ltp_mul.coalesce()

            # Clamp doesn't work on sparse tensors, so we need to break
            # it out to do it.
            indices = self.ltp_mul.indices()
            values = self.ltp_mul.values()
            torch.clamp(values, max=self.ltp_max)

            self.ltp_mul = torch.sparse_coo_tensor(
                indices, values, self.weights.shape, is_coalesced=True
            )

        return conductance, delay_buffer, new_spikes, v, refrac


# ============================================================================
# Data Utilities
# ============================================================================
class Data:
    """
    Load or build sparse weight matrix from connectivity data.
    """

    weights: torch.Tensor
    """
    The sparse weight matrix.
    Multiply by a vector of spikes to get a vector
    of how much each neuron is affected by them.
    """

    flyid2i: dict[int, int]
    """
    Maps fly neuron IDs to a continuous index.
    """

    id2flyid: np.ndarray[tuple[int], np.dtype[np.int64]]
    """
    Maps the internal neuron index back to a fly neuron ID.
    """

    coords: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    """
    Maps an internal neuron index (x, y) coordinates.
    """

    def __init__(self, device_name: str):
        wt_dir = Path(__name__).parent.parent / "data"
        connections_path = wt_dir / "connections_princeton.csv.gz"
        coordinates_path = wt_dir / "coordinates.csv.gz"

        data_conn = pd.read_csv(connections_path)
        data_coord = pd.read_csv(coordinates_path)

        all_neurons = pd.concat(
            [data_coord["root_id"], data_conn["pre_root_id"], data_conn["post_root_id"]]
        )
        all_neurons.drop_duplicates(inplace=True)
        # Cannot run inplace as it's a Series and pandas
        # wants to return a DataFrame, although we
        # need to convert it right back to a Series anyway.
        all_neurons = all_neurons.reset_index()[0]

        self.flyid2i = {v: int(k) for k, v in all_neurons.to_dict().items()}
        self.i2flyid = all_neurons.to_list()
        assert len(self.flyid2i) == len(self.i2flyid)

        num_neurons = len(self.flyid2i)

        # Extract weights.

        # From https://github.com/funkelab/drosophila_neurotransmitters
        # TODO: Fill out the zeros?
        transmitters = {"GABA": -1, "ACH": 1, "GLUT": 1, "OCT": 1, "SER": 0, "DA": 0}
        mapped_connections = data_conn["syn_count"] * data_conn["nt_type"].map(
            transmitters
        )

        self.weights = torch.sparse_coo_tensor(
            # This is ordered as [row, column].
            # Our input vector to the matrix is spike outputs (columns) and
            # the output should be
            [
                data_conn["post_root_id"].map(self.flyid2i).to_list(),
                data_conn["pre_root_id"].map(self.flyid2i).to_list(),
            ],
            mapped_connections,
            (num_neurons, num_neurons),
        ).to(dtype=torch.float32, device=device_name)

        # Extract coordinates.

        # position looks like [1 2 3] (some rows have multiple spaces)
        data_coord["position"] = data_coord["position"].str.replace("[", "")
        data_coord["position"] = data_coord["position"].str.replace("]", "")
        data_coord["position"] = data_coord["position"].str.split(" +")
        # Some

        def fix_array(arr):
            # Some entries are empty and will fail to parse
            arr = [float(i or "0") for i in arr]
            # We don't want the z coordinate for 2d plots (project downwards)
            arr = arr[:2]

            if len(arr) != 2:
                return [0.0, 0.0]
            return arr

        data_coord["position"] = data_coord["position"].apply(fix_array)

        # There are multiple positions per neuron
        data_coord.drop_duplicates(subset="root_id", keep="first", inplace=True)
        data_coord.set_index("root_id", inplace=True)

        # Pandas returns an array of lists, we need to convert it all to a single ndarray
        self.coords = np.stack(data_coord["position"].loc[self.i2flyid].to_numpy())
        assert len(self.coords) == num_neurons

        # Remap the long axis to be in the range [0, 1].
        max_coord = np.max(self.coords)
        self.coords = self.coords / max_coord


# helpers
def search_downstream(data, root, levels=3):
    downstream = set([root])
    for _ in range(levels):
        downstream_tensor = torch.tensor(list(downstream), device=data.weights.device)
        downstream = set(
            data.weights.coalesce()
            .indices()[0][
                torch.isin(data.weights.coalesce().indices()[1], downstream_tensor)
            ]
            .cpu()
        )
    return list(downstream)


# Extract cells from hierarchy
# We want to observe the change in connection between Kenyon cells and MBON cells
def extract_ids(ann):

    # Kenyon cells are the cells that get activated from a scent.
    # A given scent usually activates a unique subset of around 25 Kenyon cells.
    KC_IDS = (
        ann.loc[ann["class"].str.contains("Kenyon", na=False), "root_id"]
        .astype(np.int64)
        .values
    )

    # MBONs are the cells where memory becomes behaviorally meaningful.
    # MBONs take as input the output of Kenyon cells and have their own behavior/current change as a result.
    MBON_IDS = (
        ann.loc[ann["class"].str.contains("MBON", na=False), "root_id"]
        .astype(np.int64)
        .values
    )

    # reward dopamine, not currently used but could be
    PAM_IDS = (
        ann.loc[
            ann["class"].str.contains("PAM", na=False)
            | ann["sub_class"].str.contains("PAM", na=False),
            "root_id",
        ]
        .astype(np.int64)
        .values
    )

    # punishment dopamine, not currently used but could be
    PPL1_IDS = (
        ann.loc[
            ann["class"].str.contains("PPL1", na=False)
            | ann["sub_class"].str.contains("PPL1", na=False),
            "root_id",
        ]
        .astype(np.int64)
        .values
    )

    return KC_IDS, MBON_IDS, PAM_IDS, PPL1_IDS


# Randomly generate odor by randomly choosing a small set of Kenyon cells to activate.
def generate_odor(n_kc, active_fraction=0.05, strength=1.0):
    x = torch.zeros(n_kc)
    active = np.random.choice(n_kc, size=int(n_kc * active_fraction), replace=False)
    x[active] = strength
    return x


def get_orn_indices_for_type(data, orn_type):
    cell_types = pd.read_csv(
        Path(__name__).parent.parent / "data" / "consolidated_cell_types.csv.gz"
    )

    orn_rows = cell_types[
        cell_types.astype(str).apply(
            lambda row: row.str.contains(orn_type, na=False).any(),
            axis=1
        )
    ]

    orn_root_ids = orn_rows["root_id"].astype(np.int64).values
    orn_indices = [data.flyid2i[root_id] for root_id in orn_root_ids if root_id in data.flyid2i]

    print(f"Found {len(orn_indices)} neurons for {orn_type}.")
    return orn_indices


def build_mbon_type_indices(data, MBON_IDS, target_mbon_types):
    mbons = pd.read_csv(Path(__name__).parent.parent / "data" / "MBONs.csv")
    target_indices = []

    for root_id in MBON_IDS:
        row = mbons[mbons["root_id"] == root_id]

        if not row.empty:
            row_text = str(row.iloc[0])

            if any(mbon_type in row_text for mbon_type in target_mbon_types):
                if root_id in data.flyid2i:
                    target_indices.append(data.flyid2i[root_id])

    return target_indices


def run_trial_and_collect(model, rates, kc_indices, mbon_indices, num_steps, record_video=False, steps_per_frame=10, visualization_acc_window=50, seed=None):
    # Runs one odor trial. Returns:
    # active_kcs_seen: set of internal KC indices that spiked at least once
    # mbon_spike_sum: total spikes from selected MBONs during the trial
    # spike_sum: total spikes from all neurons during the trial
    # spike_frames: optional frames for video

    conductance, delay_buffer, spikes, v, refrac = model.state_init()

    active_kcs_seen = set()
    spike_sum = spikes.clone()
    spike_sums = []
    spike_frames = []

    kc_indices_tensor = torch.tensor(kc_indices, device=rates.device, dtype=torch.long)
    mbon_indices_tensor = torch.tensor(mbon_indices, device=rates.device, dtype=torch.long)
    mbon_spike_sum = torch.zeros(len(mbon_indices), device=rates.device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=rates.device)
        generator.manual_seed(seed)

    with torch.no_grad():
        for t_step in range(num_steps):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac, generator=generator
            )

            spike_sum += spikes

            active_kc_mask = spikes[0, kc_indices_tensor] > 0
            active_kcs_seen.update(kc_indices_tensor[active_kc_mask].cpu().tolist())
            mbon_spike_sum += spikes[0, mbon_indices_tensor]

            if record_video and t_step != 0 and t_step % steps_per_frame == 0:
                spike_sums.append(spike_sum.clone())

                frame_data = spike_sum.clone()
                frame_idx = t_step // steps_per_frame

                if frame_idx >= visualization_acc_window:
                    frame_data -= spike_sums[frame_idx - visualization_acc_window]

                spike_frames.append(frame_data.cpu())

    return active_kcs_seen, mbon_spike_sum.cpu(), spike_sum.cpu(), spike_frames


def apply_ltd_to_model_weights(model, active_kc_indices, target_mbon_indices, ltd_rate=0.05):
    # Weakens active KC -> target MBON connections.
    # KC -> MBON weights are weights[mbon_index, kc_index]
    if not active_kc_indices:
        print("No active KCs this trial; no LTD applied.")
        return

    W = model.weights.coalesce()
    indices = W.indices()
    values = W.values().clone()
    post_indices = indices[0]
    pre_indices = indices[1]

    active_kc_tensor = torch.tensor(list(active_kc_indices), device=values.device, dtype=pre_indices.dtype)
    target_mbon_tensor = torch.tensor(list(target_mbon_indices), device=values.device, dtype=post_indices.dtype)

    is_active_kc_pre = torch.isin(pre_indices, active_kc_tensor)
    is_target_mbon_post = torch.isin(post_indices, target_mbon_tensor)
    ltd_mask = is_active_kc_pre & is_target_mbon_post
    values[ltd_mask] *= 1.0 - ltd_rate
    model.weights = torch.sparse_coo_tensor(indices, values, W.shape, device=values.device).coalesce()

    print(f"Weakened {int(ltd_mask.sum().item())} KC -> MBON weights.")


def plot_things(data, show_video, mbon_ids, spike_sum, spike_frames, before, steps_per_frame, visualization_acc_window):

    had_spikes = spike_sum > 0

    if before: print(f"number of neurons that spiked before learning: {had_spikes.sum().item()}")
    if not before: print(f"number of neurons that spiked after learning: {had_spikes.sum().item()}")

    fig, ax = plt.subplots()

    plot_brain_ghost(ax, data.coords)

    if not show_video:
        mbon_indices = [data.flyid2i[mbon_id] for mbon_id in mbon_ids if mbon_id in data.flyid2i]
        mbon_had_spikes = had_spikes[mbon_indices]
        ax.scatter(
            data.coords[had_spikes, 0],
            data.coords[had_spikes, 1],
            c="darkgray",
            s=ACTIVE_NEURON_DOT_SIZE,
            alpha=0.35 * ACTIVE_NEURON_ALPHA,
            linewidths=0,
            zorder=1,
        )
        # plot the mbons that spiked with the spike_sum coloring
        ax.scatter(
            data.coords[mbon_indices, 0][mbon_had_spikes],
            data.coords[mbon_indices, 1][mbon_had_spikes],
            c=spike_sum[mbon_indices][mbon_had_spikes],
            s=HIGHLIGHT_NEURON_DOT_SIZE,
            alpha=ACTIVE_NEURON_ALPHA,
            linewidths=0,
            zorder=2,
        )
        # count of mbons total, mbons that spiked, overall neurons that spiked
        print(
            f"spiking mbons / total mbons: {mbon_had_spikes.sum().item()} / {len(mbon_indices)}, total spiking neurons: {had_spikes.sum().item()}"
        )
    else:
        if len(spike_frames) == 0:
            raise ValueError("No video frames were recorded. Try increasing t_run_sec or lowering steps_per_frame.")

        # The number of seconds we count spikes in for each spike_frame.
        rate_mul = 1 / (visualization_acc_window * steps_per_frame * DT / 1000)
        # Add 1 to prevent clipping
        max_rate = max(frame.max().item() for frame in spike_frames) * rate_mul + 1
        spike_plot = ax.scatter(
            data.coords[had_spikes, 0],
            data.coords[had_spikes, 1],
            c=spike_sum[had_spikes] * rate_mul,
            vmin=0,
            vmax=max_rate,
            s=ACTIVE_NEURON_DOT_SIZE,
            alpha=ACTIVE_NEURON_ALPHA,
            linewidths=0,
            zorder=1,
        )
        fig.colorbar(spike_plot, ax=ax)

        def show_frame(frame):
            ax.set_title(f"Brain at t = {frame * steps_per_frame * DT / 1000:.2f}s")

            frame_spike_sum = spike_frames[frame][0]
            frame_had_spikes = frame_spike_sum > 0

            spike_plot.set_offsets(data.coords[frame_had_spikes])
            spike_plot.set_array(frame_spike_sum[frame_had_spikes] * rate_mul)

            return (
                ax,
                spike_plot,
            )

        ani = animation.FuncAnimation(
            fig=fig, func=show_frame, frames=len(spike_frames), interval=DT
        )

        if before:
            ani.save("brain_before.mp4", writer=animation.FFMpegWriter(fps=10))
        else:
            ani.save("brain_after.mp4", writer=animation.FFMpegWriter(fps=10))

    plt.show()


def main():
    # parse args to get show_video
    parser = argparse.ArgumentParser(
        description="Run fly brain simulation with stimulus."
    )
    parser.add_argument(
        "--show_video",
        action="store_true",
        help="Whether to show the video of the simulation (can be slow).",
    )
    args = parser.parse_args()
    show_video = args.show_video

    t_run_sec = 0.2
    t_sim_ms = t_run_sec * 1000.0
    num_steps = int(t_sim_ms / DT)
    # How many steps should run between
    # every displayed frame.
    steps_per_frame = 10
    # How many frames to accumulate
    # for the estimated rate.
    visualization_acc_window = 50

    if torch.cuda.is_available():
        device_name = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    print(f"Using device: {device_name}")

    data = Data(device_name)
    num_neurons = data.weights.shape[0]
    print(f"Loaded {num_neurons} neurons.")

    n_run = 1

    model = TorchModel(
        n_run,
        num_neurons,
        DT,
        MODEL_PARAMS,
        data.weights,
        device=device_name,
        enable_ltp=False,
    )

    rates = torch.zeros(n_run, num_neurons, device=device_name)


    # Start olfactory learning
    classifications = pd.read_csv(Path(__name__).parent.parent / "data" / "classification.csv.gz")
    kc_ids, mbon_ids, _, _ = extract_ids(classifications)

    odor_orn_indices = get_orn_indices_for_type(data, ODOR_ORN_TYPE)

    if len(odor_orn_indices) == 0:
        raise ValueError(f"No ORNs found for {ODOR_ORN_TYPE}.")

    rates[:, odor_orn_indices] = ODOR_ORN_RATE

    print(
        f"Stimulating {ODOR_ORN_TYPE} for (+)-2-butanol "
        f"with response value {DOOR_RESPONSE_VALUE}, "
        f"simulation rate {ODOR_ORN_RATE}."
    )

    kc_indices = [data.flyid2i[kc_id] for kc_id in kc_ids if kc_id in data.flyid2i]
    mbon_indices = [data.flyid2i[mbon_id] for mbon_id in mbon_ids if mbon_id in data.flyid2i]

    if DOPAMINE_TYPE == "reward":
        target_mbon_indices = build_mbon_type_indices(data, mbon_ids, AVOIDANCE_MBON_TYPES)
    elif DOPAMINE_TYPE == "punishment":
        target_mbon_indices = build_mbon_type_indices(data, mbon_ids, APPROACH_MBON_TYPES)
    else:
        raise ValueError("DOPAMINE_TYPE must be 'reward' or 'punishment'.")

    print("Number of MBONs:", len(mbon_indices))
    print("Number of target MBONs:", len(target_mbon_indices))

    start = time.time()

    # First trial without any ltd
    pre_active_kcs, pre_mbon_spikes, pre_spike_sum, pre_spike_frames = run_trial_and_collect(
        model=model,
        rates=rates,
        kc_indices=kc_indices,
        mbon_indices=mbon_indices,
        num_steps=num_steps,
        record_video=show_video,
        steps_per_frame=steps_per_frame,
        visualization_acc_window=visualization_acc_window,
        seed=SEED,
    )

    print(f"Pre-test active KCs: {len(pre_active_kcs)}")
    print(f"Pre-test MBON spikes: {pre_mbon_spikes.sum().item()}")

    # Run 10 trials, apply ltd after each one
    for trial in range(N_TRAINING_TRIALS):
        active_kcs, mbon_spikes, train_spike_sum, train_spike_frames = run_trial_and_collect(
            model=model,
            rates=rates,
            kc_indices=kc_indices,
            mbon_indices=mbon_indices,
            num_steps=num_steps,
            seed=1000+trial
        )

        print(f"Active KCs this trial: {len(active_kcs)}")
        print(f"MBON spikes: {mbon_spikes.sum().item()}")

        apply_ltd_to_model_weights(
            model=model,
            active_kc_indices=active_kcs,
            target_mbon_indices=target_mbon_indices,
        )

    # Last trial without any ltd
    post_active_kcs, post_mbon_spikes, post_spike_sum, post_spike_frames = run_trial_and_collect(
        model=model,
        rates=rates,
        kc_indices=kc_indices,
        mbon_indices=mbon_indices,
        num_steps=num_steps,
        record_video=show_video,
        steps_per_frame=steps_per_frame,
        visualization_acc_window=visualization_acc_window,
        seed=SEED,
    )

    print(f"Post-test active KCs: {len(post_active_kcs)}")
    print(f"Post-test MBON spikes: {post_mbon_spikes.sum().item()}")

    print(f"took {time.time() - start} seconds overall")

    nonzero = (pre_mbon_spikes + post_mbon_spikes) > 0
    nonzero_indices = torch.where(nonzero)[0].numpy()

    plt.figure()
    x = np.arange(len(nonzero_indices))

    plt.bar(x - 0.2, pre_mbon_spikes[nonzero].numpy(), width=0.4, label="pre")
    plt.bar(x + 0.2, post_mbon_spikes[nonzero].numpy(), width=0.4, label="post")

    labels = [str(data.i2flyid[mbon_indices[i]]) for i in nonzero_indices]
    plt.xticks(x, labels, rotation=90)

    plt.xlabel("MBON root ID")
    plt.ylabel("Spike count during odor trial")
    plt.title("MBON response before vs after 10 LTD trials")
    plt.legend()
    plt.tight_layout()
    plt.show()

    target_mask = torch.tensor(
        [idx in set(target_mbon_indices) for idx in mbon_indices],
        dtype=torch.bool,
    )

    target_nonzero = (pre_mbon_spikes + post_mbon_spikes > 0) & target_mask

    print("Target pre MBON spikes:", pre_mbon_spikes[target_mask].sum().item())
    print("Target post MBON spikes:", post_mbon_spikes[target_mask].sum().item())

    if target_nonzero.sum().item() == 0:
        print("No target MBONs spiked in either pre or post trial.")
    else:
        target_nonzero_indices = torch.where(target_nonzero)[0].numpy()
        x = np.arange(len(target_nonzero_indices))

        plt.figure()
        plt.bar(x - 0.2, pre_mbon_spikes[target_nonzero].numpy(), width=0.4, label="pre")
        plt.bar(x + 0.2, post_mbon_spikes[target_nonzero].numpy(), width=0.4, label="post")

        labels = [str(data.i2flyid[mbon_indices[i]]) for i in target_nonzero_indices]
        plt.xticks(x, labels, rotation=90)

        plt.xlabel("Target MBON root ID")
        plt.ylabel("Spike count during odor trial")
        plt.title("Target MBON response before vs after LTD")
        plt.legend()
        plt.tight_layout()
        plt.show()
    # End olfactory learning

    plot_things(data, show_video, mbon_ids, pre_spike_sum[0], pre_spike_frames, True, steps_per_frame, visualization_acc_window)
    plot_things(data, show_video, mbon_ids, post_spike_sum[0], post_spike_frames, False, steps_per_frame, visualization_acc_window)


main()

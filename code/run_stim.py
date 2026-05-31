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
        connections_path = wt_dir / "connections_princeton.csv"
        coordinates_path = wt_dir / "coordinates.csv"

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
        transmitters = {"GABA": -1, "ACH": 1, "GLUT": 0, "OCT": 0, "SER": 0, "DA": 0}
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
    parser.add_argument(
        "--mode",  # should be either visual or olfactory
        type=str,
        default="visual",
        help="The type of stimulus to show (visual or olfactory).",
    )
    parser.add_argument(
        "--enable-ltp",
        action="store_true",
        help="Whether to enable long-term potentiation.",
    )
    args = parser.parse_args()
    show_video = args.show_video
    mode = args.mode
    enable_ltp = args.enable_ltp

    t_run_sec = 0.1
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
        enable_ltp=enable_ltp,
    )

    rates = torch.zeros(n_run, num_neurons, device=device_name)
    # TODO: Replace with a properly chosen neuron (these are all random).
    # rates[:, data.flyid2i[720575940633195148]] = 10000.0
    # rates[:, exc_indices] = stim_rate

    if mode == "visual":
        root = data.flyid2i[720575940641130368]  # visual input neuron
        downstream = search_downstream(data, root, levels=3)
        print(f"Stimulating {len(downstream)} neurons downstream of {root}.")
        rates[:, downstream] = 100.0  # some sort of low number
        rates[:, root] = 10000.0  # some sort of high number
    elif mode == "olfactory":
        classifications = pd.read_csv(
            Path(__name__).parent.parent / "data" / "classification.csv"
        )
        kc_ids, mbon_ids, _, _ = extract_ids(classifications)
        # generate a random odor by activating a random subset of Kenyon cells
        odor = generate_odor(len(kc_ids), strength=10000).to(device_name)
        rates[:, [data.flyid2i[kc_id] for kc_id in kc_ids]] = odor
    else:
        raise ValueError(f"Unknown mode: {mode}")

    conductance, delay_buffer, spikes, v, refrac = model.state_init()

    spike_sum = spikes.clone()
    # Total spikes at each timestep.
    spike_sums = []
    # Spikes in a visualization_acc_window window
    # at each timestep (effectively a constant multiple
    # of spike rate).
    spike_frames = []

    start = time.time()
    current = start
    with torch.no_grad():
        for t_step in range(num_steps):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac
            )

            spike_sum += spikes

            if t_step != 0 and t_step % steps_per_frame == 0:
                spike_sums.append(spike_sum.clone())

                frame_data = spike_sum.clone()

                frame_idx = t_step // steps_per_frame
                if frame_idx >= visualization_acc_window:
                    frame_data -= spike_sums[frame_idx - visualization_acc_window]
                spike_frames.append(frame_data)

            if t_step % (1000 * t_run_sec) == 0 and t_step != 0:
                print(
                    f"Step {t_step}/{num_steps} done, took {time.time() - current} seconds"
                )
                current = time.time()

    # Force everything onto the cpu for processing.
    spike_sum = spike_sum.cpu()
    spike_frames = [f.cpu() for f in spike_frames]

    spike_sum = spike_sum[0]
    had_spikes = spike_sum > 0
    print(f"took {time.time() - start} seconds overall")
    print(f"number of neurons that spiked: {had_spikes.sum().item()}")

    fig, ax = plt.subplots()

    plot_brain_ghost(ax, data.coords)

    if not show_video:
        if mode == "visual":  # normal
            ax.scatter(
                data.coords[had_spikes, 0],
                data.coords[had_spikes, 1],
                c=spike_sum[had_spikes],
                s=ACTIVE_NEURON_DOT_SIZE,
                linewidths=0,
                zorder=1,
            )
        if (
            mode == "olfactory"
        ):  # then we want all spikes to be a little faded, and highlight the mbons
            mbon_indices = [data.flyid2i[mbon_id] for mbon_id in mbon_ids]
            mbon_had_spikes = had_spikes[mbon_indices]
            ax.scatter(
                data.coords[had_spikes, 0],
                data.coords[had_spikes, 1],
                c="darkgray",
                alpha=0.35,
                s=ACTIVE_NEURON_DOT_SIZE,
                linewidths=0,
                zorder=1,
            )
            # plot the mbons that spiked with the spike_sum coloring
            ax.scatter(
                data.coords[mbon_indices, 0][mbon_had_spikes],
                data.coords[mbon_indices, 1][mbon_had_spikes],
                c=spike_sum[mbon_indices][mbon_had_spikes],
                s=HIGHLIGHT_NEURON_DOT_SIZE,
                linewidths=0,
                zorder=2,
            )
            # count of mbons total, mbons that spiked, overall neurons that spiked
            print(
                f"spiking mbons / total mbons: {mbon_had_spikes.sum().item()} / {len(mbon_indices)}, total spiking neurons: {had_spikes.sum().item()}"
            )
    else:
        # The number of seconds we count spikes in for each spike_frame.
        rate_mul = 1 / (visualization_acc_window * steps_per_frame * DT / 1000)
        # Add 1 to prevent clipping
        max_rate = np.max(spike_frames) * rate_mul + 1
        spike_plot = ax.scatter(
            data.coords[had_spikes, 0],
            data.coords[had_spikes, 1],
            c=spike_sum[had_spikes] * rate_mul,
            vmin=0,
            vmax=max_rate,
            s=ACTIVE_NEURON_DOT_SIZE,
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

        ani.save("brain.mp4", writer=animation.FFMpegWriter(fps=10))

    plt.show()


main()

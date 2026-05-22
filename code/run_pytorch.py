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
from pathlib import Path
import matplotlib.pyplot as plt

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

    def __init__(self, batch, size, dt, params, weights, device="cpu"):
        super().__init__()
        self.neurons = AlphaLIF(batch, size, dt, params, device=device)
        self.weights = weights.coalesce()
        self.poisson = PoissonSpikeGenerator(dt, params["scalePoisson"], device=device)
        self.scale = params["wScale"]

    def state_init(self):
        return self.neurons.state_init()

    def forward(
        self, rates, conductance, delay_buffer, spikes, v, refrac, generator=None
    ):
        spikes_input = self.poisson(rates, generator=generator)
        weighted_spikes = torch.sparse.mm(self.weights, spikes.T).T
        conductance, delay_buffer, spikes, v, refrac = self.neurons(
            self.scale * (spikes_input + weighted_spikes),
            conductance,
            delay_buffer,
            spikes,
            v,
            refrac,
        )
        return conductance, delay_buffer, spikes, v, refrac


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


def main():
    t_run_sec = 0.1
    t_sim_ms = t_run_sec * 1000.0
    num_steps = int(t_sim_ms / DT)

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
        n_run, num_neurons, DT, MODEL_PARAMS, data.weights, device=device_name
    )

    rates = torch.zeros(n_run, num_neurons, device=device_name)
    # TODO: Replace with a properly chosen neuron (these are all random).
    rates[:, data.flyid2i[720575940633195148]] = 10000.0
    # rates[:, exc_indices] = stim_rate

    conductance, delay_buffer, spikes, v, refrac = model.state_init()

    spike_sum = spikes.clone()

    # TODO: Profile this (it seems slower than expected).
    start = time.time()
    current = start
    # with torch.inference_mode():
    with torch.no_grad():
        for t_step in range(num_steps):
            conductance, delay_buffer, spikes, v, refrac = model(
                rates, conductance, delay_buffer, spikes, v, refrac
            )

            spike_sum += spikes

            if t_step % 100 == 0 and t_step != 0:
                print(
                    f"Step {t_step}/{num_steps} done, took {time.time() - current} seconds"
                )
                current = time.time()

    spike_sum = spike_sum.cpu()
    had_spikes = spike_sum[0] > 0
    print(f"took {time.time() - start} seconds overall")
    plt.scatter(
        data.coords[had_spikes, 0],
        data.coords[had_spikes, 1],
        c=spike_sum[0][had_spikes],
    )
    plt.show()


main()

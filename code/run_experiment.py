import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# To properly work, download the following files from flywire (https://codex.flywire.ai/api/download?dataset=fafb)
# and put them into the data directory. To access the link, you may need to login first:
# classification.csv.gz found in the Classification / Hierarchical Annotations section
# connections_princeton.csv.gz found in the Connections (Filtered) section

# Optionally, (but not needed for this experiment right now):
# coordinates.csv.gz found in the Marked Neuron Coordinates section
# neurons.csv.gz found in the Neurotransmitter Type Predictions section


LEARNING_RATE = 1e-4
DOPAMINE_LTD_SCALE = -1.0   # LTD (reward pathway weakening)
DOPAMINE_LTP_SCALE =  1.0   # LTP (could use for aversive pathways)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

np.random.seed(0)

# Load data
def load_data():
    ann = pd.read_csv(os.path.join(DATA_DIR, "classification.csv.gz"))
    conn = pd.read_csv(os.path.join(DATA_DIR, "connections_princeton.csv.gz"))
    return ann, conn


# Extract cells from hierarchy
# We want to observe the change in connection between Kenyon cells and MBON cells
def extract_ids(ann):

    # Kenyon cells are the cells that get activated from a scent.
    # A given scent usually activates a unique subset of around 25 Kenyon cells.
    KC_IDS = ann.loc[
        ann["class"].str.contains("Kenyon", na=False),
        "root_id"
    ].astype(np.int64).values

    # MBONs are the cells where memory becomes behaviorally meaningful.
    # MBONs take as input the output of Kenyon cells and have their own behavior/current change as a result.
    MBON_IDS = ann.loc[
        ann["class"].str.contains("MBON", na=False),
        "root_id"
    ].astype(np.int64).values

    # reward dopamine, not currently used but could be
    PAM_IDS = ann.loc[
        ann["class"].str.contains("PAM", na=False)
        | ann["sub_class"].str.contains("PAM", na=False),
        "root_id"
    ].astype(np.int64).values

    # punishment dopamine, not currently used but could be
    PPL1_IDS = ann.loc[
        ann["class"].str.contains("PPL1", na=False)
        | ann["sub_class"].str.contains("PPL1", na=False),
        "root_id"
    ].astype(np.int64).values

    return KC_IDS, MBON_IDS, PAM_IDS, PPL1_IDS


# build Kenyon cell to MBON cell connectivity matrix
def build_kc_mbon_matrix(conn, KC_IDS, MBON_IDS):
    # Find the synapses connecting KC and MBON
    kc_mbon = conn[
        conn["pre_root_id"].isin(KC_IDS)
        & conn["post_root_id"].isin(MBON_IDS)
    ].copy()

    # This obscures the real root id's for simpler graphing.
    kc_index = {nid: i for i, nid in enumerate(KC_IDS)}
    mbon_index = {nid: i for i, nid in enumerate(MBON_IDS)}

    W = np.zeros((len(KC_IDS), len(MBON_IDS)), dtype=np.float32)

    # Looks at the connections between KC and MBONs and records the number of synapses connecting them
    # Rows = KCs, Columns = MBONs, each grid cell (i,j) = num of synapses connecting the ith KC and jth MBON
    for row in kc_mbon.itertuples():
        i = kc_index[row.pre_root_id]
        j = mbon_index[row.post_root_id]
        # Could change this later to be conductance instead of raw synapse count
        W[i, j] += row.syn_count  # synapses = initial syn strength

    return torch.tensor(W)


# Randomly generate odor by randomly choosing a small set of Kenyon cells to activate.
def generate_odor(n_kc, active_fraction=0.05, strength=1.0):
    x = torch.zeros(n_kc)
    active = np.random.choice(n_kc, size=int(n_kc * active_fraction), replace=False)
    x[active] = strength
    return x


def dopamine(reward=True):
    return DOPAMINE_LTP_SCALE if reward else DOPAMINE_LTD_SCALE


class MushroomBodyModel(nn.Module):

    def __init__(self, W_kc_mbon):
        super().__init__()
        self.W = nn.Parameter(W_kc_mbon)

    def forward(self, kc_activity):
        # In the future could use the equation I = g(V-E) instead
        # E_REV = 0.0 #mV
        # V_MBON = -60.0 #mV
        # g_total = kc_activity @ self.W
        # i_syn = g_total * (E_REV - V_MBON)
        return kc_activity @ self.W  # Updates synaptic current/MBON output

    def update(self, kc, mbon, da):

        # Hebbian component: KC × MBON
        dw = torch.outer(kc, mbon)

        # dopamine modulation (3-factor rule)
        dw = da * dw

        self.W.data += LEARNING_RATE * dw


def train_step(model, kc, reward=True):
    mbon = model(kc)
    da = dopamine(reward)
    model.update(kc, mbon, da)
    return mbon.detach()


def run():
    # Right now KC is either 0 or 1 instead of firing rate (Hz) and the weights = raw synapse counts instead of conductance (nS)
    ann, conn = load_data()
    KC_IDS, MBON_IDS, PAM_IDS, PPL1_IDS = extract_ids(ann)
    W = build_kc_mbon_matrix(conn, KC_IDS, MBON_IDS)
    model = MushroomBodyModel(W)
    kc_A = generate_odor(len(KC_IDS))  # The subset of Kenyon cells a random odor activates
    pre = model(kc_A)

    for _ in range(50):
        train_step(model, kc_A, reward=True)

    post = model(kc_A)

    # each cell is one MBON neuron’s output response/resulting current to the SAME odor
    print("Pre-learning MBON activity:")
    print(pre.detach().numpy())
    print("Post-learning MBON activity:")
    print(post.detach().numpy())

    # Should probably change graph to not connect the dots because MBON cells are discrete, not continuous
    plt.xlabel("MBON root ID")
    plt.ylabel("Synpatic current")  # This is not the synaptic current quite yet, but could be the final measurement with some refinement
    plt.plot(pre.detach().numpy(), label="pre")
    plt.plot(post.detach().numpy(), label="post")
    plt.legend()
    plt.title("Synaptic current between KC and MBON cells pre vs post learning")
    plt.show()


run()
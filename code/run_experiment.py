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

# MBONs.csv is used only to figure out which MBON columns are approach/avoidance MBONs.
# This file can be found by querying for class == MBON in https://codex.flywire.ai/?dataset=fafb and downloading the CSV of the result

LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

np.random.seed(0)

AVOIDANCE_MBON_TYPES = {"MBON01", "MBON03", "MBON04", "MBON05"}
APPROACH_MBON_TYPES = {"MBON08", "MBON09", "MBON11", "MBON12", "MBON15", "MBON16", "MBON17", "MBON18","MBON19"}

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
        W[i, j] += row.syn_count  # synapses = initial syn strength = conductance

    return torch.tensor(W)


def build_mbon_type_mask(MBON_IDS, target_mbon_types):
    mbons = pd.read_csv(os.path.join(DATA_DIR, "MBONs.csv"))
    mask = torch.zeros(len(MBON_IDS))

    for j, root_id in enumerate(MBON_IDS):
        row = mbons[mbons["root_id"] == root_id]

        if not row.empty:
            row_text = str(row.iloc[0])
            if any(mbon_type in row_text for mbon_type in target_mbon_types):
                mask[j] = 1.0

    return mask


# Randomly generate odor by randomly choosing a small set of Kenyon cells to activate.
def generate_odor(n_kc, active_fraction=0.05, strength=1.0):
    x = torch.zeros(n_kc)
    active = np.random.choice(n_kc, size=int(n_kc * active_fraction), replace=False)
    x[active] = strength
    return x


class MushroomBodyModel(nn.Module):

    def __init__(self, W_kc_mbon):
        super().__init__()
        self.W = nn.Parameter(W_kc_mbon)

        # Remember which KC -> MBON connections actually exist to prevent updating nonexistent connections.
        self.connection_mask = W_kc_mbon > 0

    def forward(self, kc_activity):
        # Use the equation I = g(V-E) instead
        E_REV = 0.0 #mV
        V_MBON = -60.0 #mV
        g_total = kc_activity @ self.W
        i_syn = g_total * (E_REV - V_MBON)
        return i_syn  # Updates input synaptic current into each MBON
        # return kc_activity @ self.W

    def update(self, kc, target_mbon_mask):

        # Update active KC -> target MBON connections.
        # kc shape = [num_kc]
        # target_mbon_mask shape = [num_mbon]
        dw = torch.outer(kc, target_mbon_mask)

        # only existing KC -> MBON connections can weaken
        dw = dw * self.connection_mask

        self.W.data *= (1.0 - LEARNING_RATE * dw)
        self.W.data.clamp_(min=0.0)


def train_step(model, kc, target_mbon_mask):
    mbon = model(kc)
    model.update(kc, target_mbon_mask)
    return mbon.detach()


def run():
    # Right now KC is either 0 or 1 instead of firing rate (Hz) and the weights = raw synapse counts instead of conductance (nS)
    ann, conn = load_data()
    KC_IDS, MBON_IDS, PAM_IDS, PPL1_IDS = extract_ids(ann)
    W = build_kc_mbon_matrix(conn, KC_IDS, MBON_IDS)
    model = MushroomBodyModel(W)

    # If reward, weaken connections with avoidance MBONs.
    # If punishment, weaken connections with approach MBONs.
    reward = True
    if reward:
        target_mbon_mask = build_mbon_type_mask(MBON_IDS, AVOIDANCE_MBON_TYPES)
    else:
        target_mbon_mask = build_mbon_type_mask(MBON_IDS, APPROACH_MBON_TYPES)

    kc_A = generate_odor(len(KC_IDS))  # The subset of Kenyon cells a random odor activates
    pre = model(kc_A)

    for _ in range(1000):
        train_step(model, kc_A, target_mbon_mask)

    post = model(kc_A)

    # Print synaptic current going into targeted MBONs pre vs post learning.
    target_indices = torch.where(target_mbon_mask > 0)[0]
    print("Target MBON pre-learning activity:")
    print(pre[target_indices].detach().numpy())
    print("Target MBON post-learning activity:")
    print(post[target_indices].detach().numpy())

    plt.xlabel("MBON root ID")
    plt.ylabel("Input synaptic current into each MBON")

    # Bars instead of connected line because MBON cells are discrete, not continuous
    x = np.arange(len(pre[target_indices].detach().numpy()))
    plt.bar(x - 0.2, pre[target_indices].detach().numpy(), width=0.4, label="pre")
    plt.bar(x + 0.2, post[target_indices].detach().numpy(), width=0.4, label="post")
    plt.legend()
    plt.title("Synaptic current between KC and target MBON cells pre vs post learning")
    plt.show()


run()
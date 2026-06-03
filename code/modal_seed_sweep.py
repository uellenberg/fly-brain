import json
from collections import defaultdict
from pathlib import Path
import statistics

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_APP_DIR = "/app"
CONDITIONS = (
    ("farnesol", "reward"),
    ("farnesol", "punishment"),
    ("geosmin", "reward"),
    ("geosmin", "punishment"),
)

app = modal.App("fly-brain-seed-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "matplotlib",
        "tqdm",
    )
    .env({"MPLBACKEND": "Agg"})
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_APP_DIR,
        ignore=[
            ".git/**",
            ".venv/**",
            "__pycache__/**",
            "**/__pycache__/**",
            "*.pyc",
            "data/results/**",
        ],
    )
)


def mean_sd(values):
    if len(values) == 1:
        return statistics.mean(values), 0.0
    return statistics.mean(values), statistics.stdev(values)


def format_mean_sd(values):
    mean, sd = mean_sd(values)
    return f"{mean:.2f} +/- {sd:.2f}"


def parse_conditions(conditions: str):
    if conditions == "all":
        return CONDITIONS

    parsed_conditions = []
    for condition in conditions.split(","):
        try:
            odorant, dopamine_type = condition.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "Conditions must be 'all' or comma-separated odorant:condition pairs, "
                "for example 'farnesol:punishment,geosmin:punishment'."
            ) from exc

        parsed_condition = (odorant.strip(), dopamine_type.strip())
        if parsed_condition not in CONDITIONS:
            raise ValueError(
                f"Unknown condition {parsed_condition}. Available: {CONDITIONS}"
            )
        parsed_conditions.append(parsed_condition)

    return tuple(parsed_conditions)


@app.function(image=image, gpu="T4", timeout=7200)
def run_seed_sweep_remote(
    n_seeds: int = 10,
    seed_start: int = 0,
    max_odor_orn_rate: float = 10000.0,
    conditions: str = "all",
) -> str:
    import gc
    import os
    import sys
    import time

    os.chdir(REMOTE_APP_DIR)
    sys.path.insert(0, f"{REMOTE_APP_DIR}/code")

    import pandas as pd
    import torch

    from run_simulation import (
        APPROACH_MBON_TYPES,
        AVOIDANCE_MBON_TYPES,
        DT,
        LTD_RATE,
        MODEL_PARAMS,
        N_TRAINING_TRIALS,
        Data,
        TorchModel,
        apply_ltd_to_model_weights,
        build_mbon_type_indices,
        extract_ids,
        get_orn_indices_for_type,
        run_trial_and_collect,
        select_odorant_config,
    )

    results = []
    selected_conditions = parse_conditions(conditions)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_name}")
    print(f"Using max odor ORN rate: {max_odor_orn_rate}")

    data = Data(device_name)
    num_neurons = data.weights.shape[0]
    print(f"Loaded {num_neurons} neurons once for seed sweep.")

    classifications = pd.read_csv(f"{REMOTE_APP_DIR}/data/classification.csv.gz")
    kc_ids, mbon_ids, _, _ = extract_ids(classifications)
    kc_indices = [data.flyid2i[kc_id] for kc_id in kc_ids if kc_id in data.flyid2i]
    mbon_indices = [
        data.flyid2i[mbon_id] for mbon_id in mbon_ids if mbon_id in data.flyid2i
    ]

    target_indices_by_type = {
        "reward": build_mbon_type_indices(data, mbon_ids, AVOIDANCE_MBON_TYPES),
        "punishment": build_mbon_type_indices(data, mbon_ids, APPROACH_MBON_TYPES),
    }

    t_run_sec = 0.2
    num_steps = int((t_run_sec * 1000.0) / DT)

    for odorant, dopamine_type in selected_conditions:
        odorant_config = select_odorant_config(odorant)
        odor_orn_type = odorant_config["flywire_orn_type"]
        door_response_value = float(odorant_config["door_response_value"])
        odor_orn_rate = door_response_value * max_odor_orn_rate
        odor_orn_indices = get_orn_indices_for_type(data, odor_orn_type)
        if len(odor_orn_indices) == 0:
            raise ValueError(f"No ORNs found for {odor_orn_type}.")

        rates = torch.zeros(1, num_neurons, device=device_name)
        rates[:, odor_orn_indices] = odor_orn_rate
        target_mbon_indices = target_indices_by_type[dopamine_type]
        target_mask = torch.tensor(
            [idx in set(target_mbon_indices) for idx in mbon_indices],
            dtype=torch.bool,
        )

        print(
            f"Condition {odorant}/{dopamine_type}: {odor_orn_type}, "
            f"{len(odor_orn_indices)} ORNs, {len(target_mbon_indices)} target MBONs."
        )

        for seed in range(seed_start, seed_start + n_seeds):
            start = time.time()
            model = TorchModel(
                1,
                num_neurons,
                DT,
                MODEL_PARAMS,
                data.weights,
                device=device_name,
                enable_ltp=False,
            )

            pre_active_kcs, pre_mbon_spikes, pre_spike_sum, _ = run_trial_and_collect(
                model=model,
                rates=rates,
                kc_indices=kc_indices,
                mbon_indices=mbon_indices,
                num_steps=num_steps,
                seed=seed,
            )

            for trial in range(N_TRAINING_TRIALS):
                active_kcs, _, _, _ = run_trial_and_collect(
                    model=model,
                    rates=rates,
                    kc_indices=kc_indices,
                    mbon_indices=mbon_indices,
                    num_steps=num_steps,
                    seed=(seed + 1) * 1000 + trial,
                )
                apply_ltd_to_model_weights(
                    model=model,
                    active_kc_indices=active_kcs,
                    target_mbon_indices=target_mbon_indices,
                    ltd_rate=LTD_RATE,
                    verbose=False,
                )

            post_active_kcs, post_mbon_spikes, post_spike_sum, _ = run_trial_and_collect(
                model=model,
                rates=rates,
                kc_indices=kc_indices,
                mbon_indices=mbon_indices,
                num_steps=num_steps,
                seed=seed,
            )

            target_pre_spikes = pre_mbon_spikes[target_mask].sum().item()
            target_post_spikes = post_mbon_spikes[target_mask].sum().item()
            summary = {
                "odorant": odorant,
                "odorant_name": odorant_config["odorant_name"],
                "flywire_orn_type": odor_orn_type,
                "dopamine_type": dopamine_type,
                "seed": seed,
                "max_odor_orn_rate": max_odor_orn_rate,
                "odor_orn_rate": odor_orn_rate,
                "active_kcs_pre": len(pre_active_kcs),
                "active_kcs_post": len(post_active_kcs),
                "mbon_spikes_pre": pre_mbon_spikes.sum().item(),
                "mbon_spikes_post": post_mbon_spikes.sum().item(),
                "target_mbon_spikes_pre": target_pre_spikes,
                "target_mbon_spikes_post": target_post_spikes,
                "target_mbon_spike_delta": target_post_spikes - target_pre_spikes,
                "target_mbon_count": len(target_mbon_indices),
                "neurons_spiked_pre": int((pre_spike_sum[0] > 0).sum().item()),
                "neurons_spiked_post": int((post_spike_sum[0] > 0).sum().item()),
                "seconds": time.time() - start,
            }
            results.append(summary)
            print(
                f"{odorant}/{dopamine_type} seed={seed}: "
                f"target {summary['target_mbon_spikes_pre']:.0f} -> "
                f"{summary['target_mbon_spikes_post']:.0f} "
                f"(delta {summary['target_mbon_spike_delta']:+.0f}); "
                f"total MBON {summary['mbon_spikes_pre']:.0f} -> "
                f"{summary['mbon_spikes_post']:.0f}; "
                f"{summary['seconds']:.1f}s"
            )
            del model
            gc.collect()
            if device_name == "cuda":
                torch.cuda.empty_cache()

    grouped = defaultdict(list)
    for result in results:
        grouped[(result["odorant"], result["dopamine_type"])].append(result)

    summary_rows = []
    for odorant, dopamine_type in selected_conditions:
        group = grouped[(odorant, dopamine_type)]
        target_pre = [row["target_mbon_spikes_pre"] for row in group]
        target_post = [row["target_mbon_spikes_post"] for row in group]
        target_delta = [row["target_mbon_spike_delta"] for row in group]
        total_pre = [row["mbon_spikes_pre"] for row in group]
        total_post = [row["mbon_spikes_post"] for row in group]

        summary_rows.append(
            {
                "odorant": odorant,
                "dopamine_type": dopamine_type,
                "n": len(group),
                "target_pre_mean_sd": format_mean_sd(target_pre),
                "target_post_mean_sd": format_mean_sd(target_post),
                "target_delta_mean_sd": format_mean_sd(target_delta),
                "total_mbon_pre_mean_sd": format_mean_sd(total_pre),
                "total_mbon_post_mean_sd": format_mean_sd(total_post),
            }
        )

    print("\n| odorant | condition | n | target pre | target post | delta | total MBON pre | total MBON post |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            f"| {row['odorant']} | {row['dopamine_type']} | {row['n']} | "
            f"{row['target_pre_mean_sd']} | {row['target_post_mean_sd']} | "
            f"{row['target_delta_mean_sd']} | {row['total_mbon_pre_mean_sd']} | "
            f"{row['total_mbon_post_mean_sd']} |"
        )

    payload = json.dumps(summary_rows, sort_keys=True)
    print("\nSEED_SWEEP_SUMMARY_JSON", payload)
    return payload


@app.local_entrypoint()
def main(
    n_seeds: int = 10,
    seed_start: int = 0,
    max_odor_orn_rate: float = 10000.0,
    conditions: str = "all",
):
    run_seed_sweep_remote.remote(
        n_seeds=n_seeds,
        seed_start=seed_start,
        max_odor_orn_rate=max_odor_orn_rate,
        conditions=conditions,
    )

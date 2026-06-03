from pathlib import Path
import subprocess

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_APP_DIR = "/app"

app = modal.App("fly-brain-simulation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
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


@app.function(image=image, gpu="T4", timeout=7200)
def run_simulation_remote(
    odorant: str = "farnesol",
    dopamine_type: str = "reward",
    max_odor_orn_rate: float = 10000.0,
    show_video: bool = False,
) -> str:
    command = [
        "python",
        "code/run_simulation.py",
        "--odorant",
        odorant,
        "--dopamine_type",
        dopamine_type,
        "--max_odor_orn_rate",
        str(max_odor_orn_rate),
    ]
    if show_video:
        command.append("--show_video")

    completed = subprocess.run(
        command,
        cwd=REMOTE_APP_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Simulation failed with exit code {completed.returncode}")

    return completed.stdout


@app.local_entrypoint()
def main(
    odorant: str = "farnesol",
    dopamine_type: str = "reward",
    max_odor_orn_rate: float = 10000.0,
    show_video: bool = False,
):
    run_simulation_remote.remote(
        odorant=odorant,
        dopamine_type=dopamine_type,
        max_odor_orn_rate=max_odor_orn_rate,
        show_video=show_video,
    )

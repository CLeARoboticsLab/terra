#!/usr/bin/env python3
"""
Regenerate the dummy BC and world-model checkpoints under
examples/deploy_mppi_adaptation_loss_pred/checkpoints/.

Runs a minimal dummy-data + train pipeline (2 steps per model). Safe to re-run;
overwrites the committed example checkpoints.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CKPT_DIR = (
    REPO_ROOT / "examples" / "deploy_mppi_adaptation_loss_pred" / "checkpoints"
)

TRAIN_OVERRIDES = [
    "training.num_epochs=1",
    "training.max_steps=2",
    "training.eval_interval=1",
    "training.batch_size=4",
    "training.training_data_ratio=0.9",
    "data.num_train_workers=0",
    "data.num_test_workers=0",
    "data.prefetch_factor=null",
    "wandb.log=false",
]


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="terra_example_ckpt_") as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "dummy_data"
        ckpt_root = tmp_path / "train_ckpts"

        _run(
            [
                sys.executable,
                "scripts/generate_dummy_data.py",
                f"dataset_dir={data_dir}",
                "num_samples=64",
                "history_length=10",
                "prediction_length=10",
            ],
            REPO_ROOT,
        )

        common = [
            f"data.dataset_dir={data_dir}",
            f"checkpoint.checkpoint_dir={ckpt_root}",
            *TRAIN_OVERRIDES,
        ]

        _run(
            [
                sys.executable,
                "scripts/train.py",
                "model=behavior_cloning_with_pose_hist",
                "loss=behavior_cloning_with_pose_hist",
                "run_name=behavior_cloning",
                *common,
            ],
            REPO_ROOT,
        )

        _run(
            [
                sys.executable,
                "scripts/train.py",
                "model=world_model",
                "loss=world_model",
                "run_name=world_model",
                *common,
            ],
            REPO_ROOT,
        )

        for name in ("behavior_cloning", "world_model"):
            src = ckpt_root / name
            dst = EXAMPLE_CKPT_DIR / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    print(f"Updated example checkpoints under {EXAMPLE_CKPT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

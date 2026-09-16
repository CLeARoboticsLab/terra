#!/usr/bin/env python3
"""
Deploy a trained behavior-cloning policy with MPPI, online world-model adaptation,
and loss prediction (exploration bonus).

Load two Orbax checkpoints (BC policy + world model), construct
``MppiControllerWithAdaptation`` from ``config/deploy_config.yaml``, then run a
control / adapt loop on synthetic observations.

The default checkpoints under ``checkpoints/`` are **dummy artifacts** (two
training steps on random data). Replace them with your own trained checkpoints
for any real deployment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from terra.control.adaptation import ADAPT_PARAMS_DIR
from terra.control.mppi_controller import MppiControllerWithAdaptation
from terra.control.rewards import ErnestRewardFunction
from terra.utils import config as config_utils
from terra.utils.model import load_model_from_ckpt

EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = EXAMPLE_DIR / "config"
DEFAULT_BC_CKPT = EXAMPLE_DIR / "checkpoints" / "behavior_cloning"
DEFAULT_WM_CKPT = EXAMPLE_DIR / "checkpoints" / "world_model"
DEFAULT_DEPLOY_CONFIG = CONFIG_DIR / "deploy_config.yaml"


def _configure_jax_cache() -> None:
    user = os.environ.get("USER", "terra")
    cache_dir = Path(f"/tmp/{user}/terra_deploy_example_jax_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )


def _load_yaml_dict(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return cfg


def load_controller_configs(deploy_config: Path) -> tuple[dict, dict, dict]:
    """Return (mppi_cfg, reward_fn_cfg, adaptation_cfg) from deploy_config.yaml."""
    joint = _load_yaml_dict(deploy_config)
    for key in ("mppi", "reward_fn", "adaptation"):
        if key not in joint:
            raise ValueError(
                f"Deploy config {deploy_config} must contain top-level '{key}'."
            )
    return joint["mppi"], joint["reward_fn"], joint["adaptation"]


def _example_observation(bc_model_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a single (state, pose, action) tuple matching the system definition."""
    sys_cfg = bc_model_cfg["system"]
    obs_dim = config_utils.obs_dim_from_sys_config(sys_cfg)
    pose_dim = config_utils.pose_dim_from_sys_config(sys_cfg)
    act_dim = config_utils.acts_dim_from_sys_config(sys_cfg)

    # Replace with live ``state``, ``pose``, and ``act`` from your environment.
    state = np.zeros(obs_dim * 2, dtype=np.float32)
    pose = np.zeros(pose_dim, dtype=np.float32)
    act = np.zeros(act_dim, dtype=np.float32)
    return state, pose, act


def demonstrate_save_load_adaptation(
    controller: MppiControllerWithAdaptation,
    save_dir: Path,
) -> None:
    """
    Persist and restore online adaptation without running another control step.

    ``save_adaptation`` / ``load_adaptation`` on ``MppiControllerWithAdaptation``:

    - **Trainable state** (LoRA weights and loss-predictor parameters) under
      ``<save_dir>/adapt_params/`` via Orbax.
    - **Adaptation ring buffer** (observation / pose / action history used for
      ``adapt()``) as NumPy ``.npy`` files in ``save_dir``.

    Call ``save_adaptation`` after adaptation has run (e.g. end of an episode or
    before shutdown). Call ``load_adaptation`` before resuming control so MPPI and
    ``adapt()`` continue from the saved state.

    With ``path=None``, the controller keeps the last saved state in memory only
    (useful for quick rollback within a process).
    """
    if save_dir.exists():
        shutil.rmtree(save_dir)

    print(f"\nSaving adaptation state to {save_dir} ...")
    controller.save_adaptation(str(save_dir))

    adapt_dir = save_dir / ADAPT_PARAMS_DIR
    buffer_files = sorted(p.name for p in save_dir.glob("*.npy"))
    print(f"  wrote {adapt_dir}/ (trainable adapt parameters)")
    print(f"  wrote buffer arrays: {', '.join(buffer_files)}")

    print(f"Loading adaptation state from {save_dir} ...")
    controller.load_adaptation(str(save_dir))
    print("  loaded. The next update/run_control/adapt cycle uses this state.")

    print("\nIn-memory save/load (no path; snapshot inside the controller) ...")
    controller.save_adaptation()
    controller.load_adaptation()
    print("  restored from in-memory snapshot.")


def _check_loss_pred_outputs(out: dict, metrics: dict) -> None:
    exploration = out["reward_comps_unweighted"]["exploration_reward"]
    if not jnp.all(jnp.isfinite(exploration)):
        raise RuntimeError("Non-finite exploration_reward from loss predictor.")
    loss_pred_loss = metrics.get("losses", {}).get("loss_pred_loss")
    if loss_pred_loss is None:
        raise RuntimeError("adapt() metrics missing loss_pred_loss.")
    if not jnp.all(jnp.isfinite(jnp.asarray(loss_pred_loss))):
        raise RuntimeError("Non-finite loss_pred_loss during adaptation.")


def run_deploy_loop(
    bc_ckpt: Path,
    wm_ckpt: Path,
    mppi_cfg: dict,
    reward_fn_cfg: dict,
    adaptation_cfg: dict,
    num_steps: int,
    adapt_every_step: bool,
    adaptation_save_dir: Path,
    demo_save_load: bool,
) -> None:
    print("Loading behavior-cloning checkpoint:", bc_ckpt)
    bc_model, bc_cfg, bc_step = load_model_from_ckpt(str(bc_ckpt))
    print(f"  loaded step {bc_step}")

    print("Loading world-model checkpoint:", wm_ckpt)
    world_model, wm_cfg, wm_step = load_model_from_ckpt(str(wm_ckpt))
    print(f"  loaded step {wm_step}")

    reward_fn = ErnestRewardFunction(reward_fn_cfg)

    print("Building MppiControllerWithAdaptation (wraps world model + loss predictor)...")
    controller = MppiControllerWithAdaptation(
        bc_model,
        bc_cfg,
        world_model,
        wm_cfg,
        mppi_cfg,
        reward_fn,
        adaptation_cfg,
    )

    state, pose, act = _example_observation(bc_cfg)
    print("Initializing controller (JIT compile + buffer fill; may take a minute)...")
    controller.initialize(state, pose, act)

    print(f"Running {num_steps} control steps...")
    start = time.time()
    last_out = None
    last_metrics: dict = {}
    for step in range(num_steps):
        controller.update(state, pose, act)
        planner_poses = controller._make_dummy_planner_poses()
        last_out = controller.run_control(planner_poses)
        if adapt_every_step:
            last_metrics = controller.adapt(compute_metrics=True)
        if step == 0 or step == num_steps - 1:
            # ``actions`` is the full MPPI horizon (T, act_dim); index 0 is the command to apply now.
            action_now = np.asarray(last_out["actions"][0])
            exploration = float(
                last_out["reward_comps_unweighted"]["exploration_reward"]
            )
            print(
                f"  step {step}: action_now={action_now}, "
                f"exploration_reward={exploration:.4f}"
            )

    elapsed = time.time() - start
    print(f"Average step time: {elapsed / num_steps * 1000:.2f} ms")

    if adapt_every_step and last_out is not None:
        _check_loss_pred_outputs(last_out, last_metrics)
        print("Loss-prediction outputs look valid (finite exploration + loss_pred_loss).")

    if demo_save_load:
        demonstrate_save_load_adaptation(controller, adaptation_save_dir)

    print("Done.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=DEFAULT_BC_CKPT,
        help="Orbax checkpoint directory for the BC / policy model.",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        type=Path,
        default=DEFAULT_WM_CKPT,
        help="Orbax checkpoint directory for the pretrained world model.",
    )
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=DEFAULT_DEPLOY_CONFIG,
        help="YAML with top-level mppi, reward_fn, and adaptation sections.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="Number of control/adapt iterations to run after initialize().",
    )
    parser.add_argument(
        "--no-adapt",
        action="store_true",
        help="Run control only (skip online adaptation updates).",
    )
    parser.add_argument(
        "--adaptation-save-dir",
        type=Path,
        default=EXAMPLE_DIR / ".adaptation_state",
        help="Directory for the save/load adaptation demo (disk checkpoint).",
    )
    parser.add_argument(
        "--no-adaptation-save-demo",
        action="store_true",
        help="Skip the save_adaptation / load_adaptation demonstration at the end.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_jax_cache()
    args = parse_args(argv)

    for path, label in [
        (args.bc_checkpoint, "BC checkpoint"),
        (args.world_model_checkpoint, "world-model checkpoint"),
    ]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            return 1

    if not args.deploy_config.exists():
        print(f"Error: deploy config not found: {args.deploy_config}", file=sys.stderr)
        return 1

    mppi_cfg, reward_fn_cfg, adaptation_cfg = load_controller_configs(
        args.deploy_config
    )

    run_deploy_loop(
        args.bc_checkpoint,
        args.world_model_checkpoint,
        mppi_cfg,
        reward_fn_cfg,
        adaptation_cfg,
        args.num_steps,
        adapt_every_step=not args.no_adapt,
        adaptation_save_dir=args.adaptation_save_dir,
        demo_save_load=not args.no_adaptation_save_demo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

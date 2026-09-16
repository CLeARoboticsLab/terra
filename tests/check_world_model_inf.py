from terra.utils.model import load_model_from_ckpt
import terra.utils.config as config_utils
import jax
import time
import argparse


DEFAULT_CHECKPOINT = "/data/ernest_darts_rl/checkpoints/2025-07-27_world_model_ernest_no-vel_step_ripples_0.5step-per"


def main(args):
    NUM_INFS = 100
    B = 1000

    model, cfg, _ = load_model_from_ckpt(args.checkpoint)
    model_fn = jax.jit(model.inference)

    H = cfg["model"]["dataset"]["history_length"]
    T = cfg["model"]["dataset"]["prediction_length"]
    obs_dim = config_utils.obs_dim_from_sys_config(cfg["system"])
    pose_dim = config_utils.pose_dim_from_sys_config(cfg["system"])
    acts_dim = config_utils.acts_dim_from_sys_config(cfg["system"])

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    obs_hist = jax.random.normal(keys[0], (B, H + 1, obs_dim))
    pose_hist = jax.random.normal(keys[1], (B, H + 1, pose_dim))
    acts_hist = jax.random.normal(keys[2], (B, H, acts_dim))
    fut_acts = jax.random.normal(keys[3], (B, T, acts_dim))

    model_in = {
        "obs_hist": obs_hist,
        "pose_hist": pose_hist,
        "acts_hist": acts_hist,
        "fut_acts": fut_acts,
    }

    # Run once to jit
    model_fn(model_in)

    start_t = time.time()
    for _ in range(NUM_INFS):
        jax.block_until_ready(model_fn(model_in))
    end_t = time.time()

    print(f"Inference rate: {NUM_INFS / (end_t - start_t):.2f} Hz")
    print(f"Time per prediction: {(end_t - start_t) / (NUM_INFS * B) * 1000000:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check world model inference speed."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )
    args = parser.parse_args()
    main(args)

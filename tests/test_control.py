import os
import getpass
import jax

user = os.environ.get("USER", getpass.getuser())
jax.config.update("jax_compilation_cache_dir", f"/tmp/{user}/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

from terra.control.controller import Controller
from terra.control.mppi_controller import (
    MppiState,
    MppiController,
    MppiControllerWithAdaptation,
)
from terra.control.rewards import ErnestRewardFunction
from terra.modeling.models import ModelBase, get_model
import terra.utils.model as model_utils
import terra.utils.config as config_utils
import flax.nnx as nnx
import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf
import numpy as np
import time
import jax.numpy as jnp
import jax
import shutil

REL_CONFIG_PATH = "../config"
NUM_STEPS = 100
ADAPT_CKPT_PATH = "/tmp/test_adapt_checkpoint"


@pytest.fixture
def config(request):
    model_name = request.param
    return _get_config(model_name)


@pytest.mark.parametrize(
    "config",
    ["behavior_cloning_with_pose_hist"],
    indirect=True,
)
def test_controller(config: dict):
    model_type = config["model"]["type"]
    state, pose, act = _dummy_data(config)

    model = _make_dummy_model(config)
    controller = Controller(model, config)
    controller.initialize(state, pose, act)

    start_time = time.time()
    for _ in range(NUM_STEPS):
        controller.update(state, pose, act)
        planner_poses = controller._make_dummy_planner_poses()
        _ = controller.run_control(planner_poses)
    time_per_step = (time.time() - start_time) / NUM_STEPS
    print(f"Maximum controller frequency for {model_type}: {1 / time_per_step:.2f} Hz")


def test_mppi_controller():
    bc_config = _get_config("behavior_cloning_with_pose_hist")
    world_model_config = _get_config("world_model")
    state, pose, act = _dummy_data(bc_config)

    bc_model = _make_dummy_model(bc_config)
    world_model = _make_dummy_model(world_model_config)
    reward_fn = ErnestRewardFunction({})
    controller = MppiController(
        bc_model, bc_config, world_model, world_model_config, {}, reward_fn
    )
    controller.initialize(state, pose, act)

    start_time = time.time()
    for _ in range(NUM_STEPS):
        controller.update(state, pose, act)
        planner_poses = controller._make_dummy_planner_poses()
        _ = controller.run_control(planner_poses)
        _ = controller.compute_current_reward()
    time_per_step = (time.time() - start_time) / NUM_STEPS
    print(
        f"Maximum controller frequency for mppi_controller: {1 / time_per_step:.2f} Hz"
    )


def test_mppi_controller_adaptation():
    adaptation_config = {
        "buffer_length": 40,
        "lora": True,
        "lora_rank": 8,
        "lora_add_to": ("mlp",),
        "learning_rate": 1e-3,
        "batch_size": 4,
        "control_steps_before_adapt": 0,
        "always_include_latest": 1,
        "loss": {
            "type": "world_model_adaptation",
            "weights": {
                "norm_deltaW_loss": 1.0,
                "obs_loss": 1.0,
                "poses_loss": 1.0,
                "reward_scaled_loss": 1.0,
            },
            "component_weights": {
                "poses": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
        },
    }

    bc_config = _get_config("behavior_cloning_with_pose_hist")
    world_model_config = _get_config("world_model")
    state, pose, act = _dummy_data(bc_config)

    bc_model = _make_dummy_model(bc_config)
    world_model = _make_dummy_model(world_model_config)
    reward_fn = ErnestRewardFunction({})
    controller = MppiControllerWithAdaptation(
        bc_model,
        bc_config,
        world_model,
        world_model_config,
        {},
        reward_fn,
        adaptation_config,
    )
    controller.initialize(state, pose, act)

    def controller_step(adapt=True):
        controller.update(state, pose, act)
        planner_poses = controller._make_dummy_planner_poses()
        out = controller.run_control(planner_poses)
        if adapt:
            metrics = controller.adapt(compute_metrics=True)
        else:
            metrics = {}
        return out, metrics

    start_time = time.time()
    for _ in range(NUM_STEPS):
        # st = time.time()
        out, metrics = controller_step()
        # print("Adaptation metrics:", metrics)
        # print(f"Step time: {(time.time()-st)*1000:.4f} ms")
    time_per_step = (time.time() - start_time) / NUM_STEPS
    print(
        f"Maximum controller frequency for mppi_controller "
        f"with adaptation: {1 / time_per_step:.2f} Hz"
    )

    # Without adaptation, the controller should produce the same output
    controller.mppi_state = MppiState(
        mean=jnp.zeros((controller.T, controller.act_dim)),
        std=jnp.ones((controller.T, controller.act_dim)),
        key_data=controller._to_key_data(jax.random.PRNGKey(0)),
    )
    out1, _ = controller_step(adapt=False)
    controller.mppi_state = MppiState(
        mean=jnp.zeros((controller.T, controller.act_dim)),
        std=jnp.ones((controller.T, controller.act_dim)),
        key_data=controller._to_key_data(jax.random.PRNGKey(0)),
    )
    out2, _ = controller_step(adapt=False)
    assert jnp.all(jnp.equal(out1["predicted_poses"], out2["predicted_poses"]))

    # With adaptation, the controller should produce different outputs
    controller.mppi_state = MppiState(
        mean=jnp.zeros((controller.T, controller.act_dim)),
        std=jnp.ones((controller.T, controller.act_dim)),
        key_data=controller._to_key_data(jax.random.PRNGKey(0)),
    )
    out1, _ = controller_step(adapt=True)
    controller.mppi_state = MppiState(
        mean=jnp.zeros((controller.T, controller.act_dim)),
        std=jnp.ones((controller.T, controller.act_dim)),
        key_data=controller._to_key_data(jax.random.PRNGKey(0)),
    )
    out2, metrics = controller_step(adapt=True)
    print("Adaptation metrics:", metrics)
    assert not jnp.all(jnp.equal(out1["predicted_poses"], out2["predicted_poses"]))


def test_mppi_controller_adaptation_with_loss_pred():
    adaptation_config = {
        "buffer_length": 400,
        "lora": True,
        "lora_rank": 8,
        "lora_add_to": ("mlp",),
        "learning_rate": 1e-3,
        "batch_size": 4,
        "control_steps_before_adapt": 0,
        "always_include_latest": 1,
        "seed": 0,
        "loss": {
            "type": "world_model_adaptation",
            "weights": {
                "norm_deltaW_loss": 1.0,
                "obs_loss": 1.0,
                "poses_loss": 1.0,
                "reward_scaled_loss": 1.0,
            },
            "component_weights": {
                "poses": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
        },
    }
    adaptation_config["loss_pred"] = {
        "use_loss_pred": True,
        "thresholds": [0.5],
        "class_weights": [0.5, 1.0],
        "init": {
            "type": "highest_loss_bin",
            "prior_prob": 0.99,
        },
        "loss_pred_loss_weight": 1.0,
        "attention": {
            "layers": 2,
            "heads": 4,
            "mlp_hidden_size_factor": 2,
            "mask": "causal",
        },
        "decoder": {
            "layers": 2,
            "mlp_hidden_size_factor": 1,
        },
    }

    bc_config = _get_config("behavior_cloning_with_pose_hist")
    world_model_config = _get_config("world_model")
    state, pose, act = _dummy_data(bc_config)

    bc_model = _make_dummy_model(bc_config)
    world_model = _make_dummy_model(world_model_config)
    reward_fn = ErnestRewardFunction({})
    controller = MppiControllerWithAdaptation(
        bc_model,
        bc_config,
        world_model,
        world_model_config,
        {},
        reward_fn,
        adaptation_config,
    )
    controller.initialize(state, pose, act)

    def controller_step(adapt=True):
        controller.update(state, pose, act)
        planner_poses = controller._make_dummy_planner_poses()
        out = controller.run_control(planner_poses)
        if adapt:
            metrics = controller.adapt(compute_metrics=True)
        else:
            metrics = {}
        return out, metrics

    def check_loss_pred_outputs(out, metrics):
        assert "exploration_reward" in out["reward_comps_unweighted"]
        exploration_reward = out["reward_comps_unweighted"]["exploration_reward"]
        assert jnp.all(jnp.isfinite(exploration_reward))
        assert "loss_pred_loss" in metrics["losses"]
        assert jnp.all(jnp.isfinite(jnp.asarray(metrics["losses"]["loss_pred_loss"])))

    start_time = time.time()
    for _ in range(NUM_STEPS):
        # st = time.time()
        out, metrics = controller_step()
        # print("Adaptation metrics:", metrics)
        # print(f"Step time: {(time.time()-st)*1000:.4f} ms")
    check_loss_pred_outputs(out, metrics)
    time_per_step = (time.time() - start_time) / NUM_STEPS
    print(
        f"Maximum controller frequency for mppi_controller "
        f"with adaptation and loss_pred: {1 / time_per_step:.2f} Hz"
    )

    controller.save_adaptation()
    controller.load_adaptation()

    start_time = time.time()
    for _ in range(NUM_STEPS):
        out, metrics = controller_step()
    check_loss_pred_outputs(out, metrics)
    time_per_step = (time.time() - start_time) / NUM_STEPS
    print(
        f"Maximum controller frequency for mppi_controller "
        f"with adaptation and loss_pred, saved and loaded: {1 / time_per_step:.2f} Hz"
    )

    if os.path.exists(ADAPT_CKPT_PATH):
        shutil.rmtree(ADAPT_CKPT_PATH)
    controller.save_adaptation(ADAPT_CKPT_PATH)
    controller.load_adaptation(ADAPT_CKPT_PATH)

    start_time = time.time()
    for _ in range(NUM_STEPS):
        out, metrics = controller_step()
    check_loss_pred_outputs(out, metrics)
    time_per_step = (time.time() - start_time) / NUM_STEPS
    print(
        f"Maximum controller frequency for mppi_controller "
        f"with adaptation and loss_pred, saved and loaded (checkpoint given): {1 / time_per_step:.2f} Hz"
    )


def _get_config(model_name: str) -> dict:
    with initialize(config_path=REL_CONFIG_PATH, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"model={model_name}", f"loss={model_name}"],
        )
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return cfg


def _dummy_data(config: dict):
    state = np.ones(
        config_utils.obs_dim_from_sys_config(config["system"]) * 2, dtype=np.float32
    )
    pose = np.ones(
        config_utils.pose_dim_from_sys_config(config["system"]),
        dtype=np.float32,
    )
    act = np.ones(
        config_utils.acts_dim_from_sys_config(config["system"]), dtype=np.float32
    )
    return state, pose, act


def _make_dummy_model(config: dict) -> ModelBase:
    obs_dim = config_utils.obs_dim_from_sys_config(config["system"])
    pose_dim = config_utils.pose_dim_from_sys_config(config["system"])
    act_dim = config_utils.acts_dim_from_sys_config(config["system"])
    scaler_params_struct = {
        "obs": obs_dim,
        "poses": pose_dim,
        "acts": act_dim,
    }
    config["scaler_params_struct"] = scaler_params_struct
    dummy_scaler_params = model_utils.make_dummy_scaler_params(config)
    model = get_model(config, dummy_scaler_params, nnx.Rngs(0))
    return model


if __name__ == "__main__":
    test_mppi_controller_adaptation_with_loss_pred()

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

from terra.control import sensor_noise as sn
from terra.control.mppi_controller import (
    MppiController,
    MppiControllerWithAdaptation,
)
from terra.control.rewards import ErnestRewardFunction
from terra.utils.buffer import MultiRingBuffer
import terra.utils.config as config_utils
import terra.utils.model as model_utils
import jax.numpy as jnp
import numpy as np
import pytest

from test_control import _get_config, _dummy_data, _make_dummy_model


# Small MPPI settings: these tests exercise measurement ingestion, not
# planning quality, so keep the JIT/rollout work as cheap as possible.
FAST_MPPI_CFG = {
    "num_samples": 20,
    "num_elites": 5,
    "iterations": 1,
    "mixture_coef": 0.5,
}

ADAPTATION_CFG = {
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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _NoiseMirror:
    """Reproduces the injector's draws so tests can predict exact values."""

    def __init__(self, seed, level, obs_std, pose_std, yaw_idx):
        self.rng = np.random.default_rng(seed)
        self.obs_noise_std = level * np.asarray(obs_std, dtype=np.float64)
        self.pose_noise_std = level * np.asarray(pose_std, dtype=np.float64)
        self.yaw_idx = yaw_idx

    def next(self, clean_obs, clean_pose):
        obs = clean_obs + self.rng.standard_normal(
            self.obs_noise_std.shape
        ) * self.obs_noise_std
        pose = clean_pose + self.rng.standard_normal(
            self.pose_noise_std.shape
        ) * self.pose_noise_std
        pose = pose.copy()
        pose[self.yaw_idx] = sn.wrap_angle(pose[self.yaw_idx])
        return obs, pose


def _dims(config):
    sys_cfg = config["system"]
    return (
        config_utils.obs_dim_from_sys_config(sys_cfg),
        config_utils.pose_dim_from_sys_config(sys_cfg),
        sys_cfg["poses"]["se2_idxs"]["yaw"],
    )


def _set_scaler_stds(model, obs_std=None, pose_std=None):
    """Overrides the model's dataset standard deviations in place."""
    if obs_std is not None:
        model.scaler.scaler_params["obs"]["std"].value = jnp.asarray(
            obs_std, dtype=jnp.float32
        )
    if pose_std is not None:
        model.scaler.scaler_params["poses"]["std"].value = jnp.asarray(
            pose_std, dtype=jnp.float32
        )


def _make_controller(
    sensor_noise_cfg=None,
    obs_std=None,
    pose_std=None,
    bc_name="behavior_cloning_with_pose_hist",
    world_model_name="world_model",
    cls=MppiController,
    extra_args=(),
):
    bc_config = _get_config(bc_name)
    world_model_config = _get_config(world_model_name)
    bc_model = _make_dummy_model(bc_config)
    world_model = _make_dummy_model(world_model_config)
    _set_scaler_stds(world_model, obs_std, pose_std)
    controller = cls(
        bc_model,
        bc_config,
        world_model,
        world_model_config,
        FAST_MPPI_CFG,
        ErnestRewardFunction({}),
        *extra_args,
        sensor_noise_cfg=sensor_noise_cfg,
    )
    return controller, bc_config


def _clean_item(config, state, system_pose):
    sys_cfg = config["system"]
    return (
        model_utils.obs_from_state(state, sys_cfg).astype(np.float64),
        model_utils.pose_from_system_pose(system_pose, sys_cfg).astype(np.float64),
    )


def _clean_buf_item(config, state, system_pose, act):
    """The distilled item a controller builds before any perturbation."""
    sys_cfg = config["system"]
    return {
        "obs": model_utils.obs_from_state(state, sys_cfg),
        "pose": model_utils.pose_from_system_pose(system_pose, sys_cfg),
        "act": act,
    }


def _item_at(hist, i):
    return {"obs": hist["obs"][i], "pose": hist["pose"][i], "act": hist["act"][i]}


def _reward_from_items(controller, prev_item, cur_item):
    """Reproduces `compute_current_reward()` from an explicit transition.

    Only the *source* of the measurements differs from the controller's own
    path, so this pins down which buffer the actual reward is read from.
    """
    model_inputs = {
        "obs_hist": prev_item["obs"][None],
        "pose_hist": prev_item["pose"][None],
        "acts_hist": prev_item["act"][None],
        "fut_acts": cur_item["act"][None],
    }
    model_outputs = {
        "obs": cur_item["obs"][None],
        "poses": cur_item["pose"][None],
    }
    model_inputs = jax.tree.map(controller._to_dev_f32, model_inputs)
    model_inputs = model_utils.repeat_along_batch_dim(model_inputs, 1)
    model_outputs = jax.tree.map(controller._to_dev_f32, model_outputs)
    model_outputs = model_utils.repeat_along_batch_dim(model_outputs, 1)

    reward, comps, comps_unweighted = controller._calc_reward(
        controller.path_line_def, model_inputs, model_outputs
    )
    return jax.tree.map(
        controller._to_scalar,
        {
            "reward": reward,
            "reward_comps": comps,
            "reward_comps_unweighted": comps_unweighted,
        },
    )


def _assert_reward_equal(actual, expected):
    np.testing.assert_allclose(actual["reward"], expected["reward"], rtol=1e-6)
    for key in ("reward_comps", "reward_comps_unweighted"):
        assert set(actual[key]) == set(expected[key])
        for name, value in expected[key].items():
            np.testing.assert_allclose(actual[key][name], value, rtol=1e-6)


def _shifted_data(config, scale, offset):
    state, pose, act = _dummy_data(config)
    return (
        (state * scale + offset).astype(np.float32),
        (pose * scale - offset).astype(np.float32),
        (act * scale).astype(np.float32),
    )


def _adapted_noisy_controller(seed=21, level=0.5):
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, _ = _dims(config_probe)
    return _make_controller(
        sensor_noise_cfg={"enabled": True, "level": level, "seed": seed},
        obs_std=np.ones(obs_dim),
        pose_std=np.ones(pose_dim),
        cls=MppiControllerWithAdaptation,
        extra_args=(dict(ADAPTATION_CFG),),
    )


# --------------------------------------------------------------------------
# 1. Disabled / absent config keeps the legacy path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sensor_noise_cfg", [None, {}, {"enabled": False}])
def test_disabled_config_uses_legacy_path(sensor_noise_cfg):
    controller, config = _make_controller(sensor_noise_cfg=sensor_noise_cfg)
    state, pose, act = _dummy_data(config)
    controller.initialize(state, pose, act)
    for _ in range(3):
        controller.update(state, pose, act)

    assert controller.sensor_noise is None
    clean_obs, clean_pose = _clean_item(config, state, pose)
    hist = controller.buf.get()
    np.testing.assert_array_equal(hist["obs"], np.tile(clean_obs, (len(hist["obs"]), 1)))
    np.testing.assert_array_equal(
        hist["pose"], np.tile(clean_pose, (len(hist["pose"]), 1))
    )


def test_disabled_config_never_inspects_scaler():
    # A disabled config must not touch scaler data at all, so a model that
    # cannot provide any is still fine. `self` is unused on the disabled path.
    make = MppiController._make_sensor_noise_injector
    assert make(None, {"enabled": False}, object(), None) is None
    assert make(None, None, object(), None) is None
    assert (
        sn.make_sensor_noise_injector(
            {"enabled": False, "level": "nonsense"}, {}, 3, 3, 0
        )
        is None
    )


def test_disabled_adapted_controller_uses_legacy_path():
    controller, config = _make_controller(
        cls=MppiControllerWithAdaptation, extra_args=(dict(ADAPTATION_CFG),)
    )
    state, pose, act = _dummy_data(config)
    controller.initialize(state, pose, act)
    controller.update(state, pose, act)

    assert controller.sensor_noise is None
    clean_obs, clean_pose = _clean_item(config, state, pose)
    for hist in (controller.buf.get(), controller.adapt_buf.get()):
        np.testing.assert_array_equal(
            hist["obs"], np.tile(clean_obs, (len(hist["obs"]), 1))
        )
        np.testing.assert_array_equal(
            hist["pose"], np.tile(clean_pose, (len(hist["pose"]), 1))
        )


# --------------------------------------------------------------------------
# 2-6. Enabled behavior on the standard MPPI controller
# --------------------------------------------------------------------------


def test_enabled_noise_is_deterministic_and_reproducible():
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, yaw_idx = _dims(config_probe)
    obs_std = np.arange(1, obs_dim + 1, dtype=np.float64) * 0.1
    pose_std = np.arange(1, pose_dim + 1, dtype=np.float64) * 0.2
    cfg = {"enabled": True, "level": 0.3, "seed": 7}

    controller, config = _make_controller(
        sensor_noise_cfg=cfg, obs_std=obs_std, pose_std=pose_std
    )
    state, pose, act = _dummy_data(config)
    clean_obs, clean_pose = _clean_item(config, state, pose)

    mirror = _NoiseMirror(7, 0.3, obs_std, pose_std, yaw_idx)
    controller.initialize(state, pose, act)  # reset draws one realization
    exp_obs, exp_pose = mirror.next(clean_obs, clean_pose)
    hist = controller.buf.get()
    for i in range(len(hist["obs"])):
        np.testing.assert_allclose(hist["obs"][i], exp_obs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(hist["pose"][i], exp_pose, rtol=1e-5, atol=1e-6)

    # 6a. Consecutive updates advance the generator.
    seen_obs = []
    for _ in range(3):
        controller.update(state, pose, act)
        exp_obs, exp_pose = mirror.next(clean_obs, clean_pose)
        hist = controller.buf.get()
        np.testing.assert_allclose(hist["obs"][-1], exp_obs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(hist["pose"][-1], exp_pose, rtol=1e-5, atol=1e-6)
        seen_obs.append(np.array(hist["obs"][-1]))
    assert not np.allclose(seen_obs[0], seen_obs[1])

    # 6b. reset() does not restart the sequence.
    controller.reset(state, pose, act)
    exp_obs, exp_pose = mirror.next(clean_obs, clean_pose)
    hist = controller.buf.get()
    np.testing.assert_allclose(hist["obs"][-1], exp_obs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hist["pose"][-1], exp_pose, rtol=1e-5, atol=1e-6)
    assert not np.allclose(hist["obs"][-1], seen_obs[0])

    # 2b. A second controller with the same seed reproduces the same draws.
    controller2, config2 = _make_controller(
        sensor_noise_cfg=cfg, obs_std=obs_std, pose_std=pose_std
    )
    controller2.initialize(state, pose, act)
    controller2.update(state, pose, act)
    mirror2 = _NoiseMirror(7, 0.3, obs_std, pose_std, yaw_idx)
    mirror2.next(clean_obs, clean_pose)
    exp_obs, exp_pose = mirror2.next(clean_obs, clean_pose)
    hist2 = controller2.buf.get()
    np.testing.assert_allclose(hist2["obs"][-1], exp_obs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hist2["pose"][-1], exp_pose, rtol=1e-5, atol=1e-6)


def test_noise_scale_is_level_times_dataset_std_and_zeros_are_noiseless():
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, yaw_idx = _dims(config_probe)
    obs_std = np.ones(obs_dim, dtype=np.float64)
    obs_std[0] = 0.0
    pose_std = np.ones(pose_dim, dtype=np.float64) * 2.0
    pose_std[1] = 0.0
    cfg = {"enabled": True, "level": 0.5, "seed": 3}

    controller, config = _make_controller(
        sensor_noise_cfg=cfg, obs_std=obs_std, pose_std=pose_std
    )
    injector = controller.sensor_noise
    np.testing.assert_allclose(injector.obs_noise_std, 0.5 * obs_std)
    np.testing.assert_allclose(injector.pose_noise_std, 0.5 * pose_std)

    state, pose, act = _dummy_data(config)
    clean_obs, clean_pose = _clean_item(config, state, pose)
    controller.initialize(state, pose, act)
    for _ in range(5):
        controller.update(state, pose, act)

    hist = controller.buf.get()
    # Zero-std features stay exactly at the clean measurement.
    np.testing.assert_allclose(hist["obs"][:, 0], clean_obs[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hist["pose"][:, 1], clean_pose[1], rtol=1e-5, atol=1e-6)
    # Non-zero-std features moved.
    assert not np.allclose(hist["obs"][:, 1], clean_obs[1])


def test_inputs_are_not_mutated():
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, _ = _dims(config_probe)
    cfg = {"enabled": True, "level": 1.0, "seed": 11}
    controller, config = _make_controller(
        sensor_noise_cfg=cfg,
        obs_std=np.ones(obs_dim),
        pose_std=np.ones(pose_dim),
    )
    state, pose, act = _dummy_data(config)
    state_ref, pose_ref, act_ref = state.copy(), pose.copy(), act.copy()

    controller.initialize(state, pose, act)
    controller.update(state, pose, act)
    controller.reset(state, pose, act)

    np.testing.assert_array_equal(state, state_ref)
    np.testing.assert_array_equal(pose, pose_ref)
    np.testing.assert_array_equal(act, act_ref)


def test_yaw_is_wrapped_and_other_features_are_not_clipped():
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, yaw_idx = _dims(config_probe)
    obs_std = np.ones(obs_dim, dtype=np.float64) * 50.0
    pose_std = np.ones(pose_dim, dtype=np.float64) * 50.0
    cfg = {"enabled": True, "level": 1.0, "seed": 5}

    controller, config = _make_controller(
        sensor_noise_cfg=cfg, obs_std=obs_std, pose_std=pose_std
    )
    state, pose, act = _dummy_data(config)
    clean_obs, clean_pose = _clean_item(config, state, pose)
    mirror = _NoiseMirror(5, 1.0, obs_std, pose_std, yaw_idx)

    controller.initialize(state, pose, act)
    mirror.next(clean_obs, clean_pose)
    for _ in range(10):
        controller.update(state, pose, act)
        exp_obs, exp_pose = mirror.next(clean_obs, clean_pose)
        hist = controller.buf.get()
        np.testing.assert_allclose(hist["pose"][-1], exp_pose, rtol=1e-5, atol=1e-6)

    hist = controller.buf.get()
    yaw = hist["pose"][:, yaw_idx]
    assert np.all(yaw >= -np.pi - 1e-5) and np.all(yaw <= np.pi + 1e-5)
    # Nothing else is clipped: with std 50 some feature must leave [-pi, pi].
    other_pose_idxs = [i for i in range(pose_dim) if i != yaw_idx]
    assert np.any(np.abs(hist["pose"][:, other_pose_idxs]) > np.pi)
    assert np.any(np.abs(hist["obs"]) > np.pi)


# --------------------------------------------------------------------------
# 7. Adapted MPPI shares one realization between both buffers
# --------------------------------------------------------------------------


def test_adapted_controller_shares_one_realization():
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, _ = _dims(config_probe)
    cfg = {"enabled": True, "level": 0.5, "seed": 13}

    controller, config = _make_controller(
        sensor_noise_cfg=cfg,
        obs_std=np.ones(obs_dim),
        pose_std=np.ones(pose_dim),
        cls=MppiControllerWithAdaptation,
        extra_args=(dict(ADAPTATION_CFG),),
    )
    state, pose, act = _dummy_data(config)
    clean_obs, _ = _clean_item(config, state, pose)

    controller.initialize(state, pose, act)
    assert controller.sensor_noise is not None
    assert controller._pending_adapt_fill is False

    # Initialization: both buffers hold the same single noisy realization.
    ctrl_hist = controller.buf.get()
    adapt_hist = controller.adapt_buf.get()
    np.testing.assert_array_equal(ctrl_hist["obs"][-1], adapt_hist["obs"][-1])
    np.testing.assert_array_equal(ctrl_hist["pose"][-1], adapt_hist["pose"][-1])
    assert not np.allclose(ctrl_hist["obs"][-1], clean_obs)

    # Updates: identical values land in both buffers.
    for _ in range(3):
        controller.update(state, pose, act)
        ctrl_hist = controller.buf.get()
        adapt_hist = controller.adapt_buf.get()
        np.testing.assert_array_equal(ctrl_hist["obs"][-1], adapt_hist["obs"][-1])
        np.testing.assert_array_equal(ctrl_hist["pose"][-1], adapt_hist["pose"][-1])
    assert not np.allclose(ctrl_hist["obs"][-1], ctrl_hist["obs"][-2])

    # Episode-boundary markers are preserved.
    assert not controller.adapt_buf.episode_starts.get()[-1]
    controller.reset(state, pose, act)
    assert controller.was_reset
    controller.update(state, pose, act)
    assert controller.adapt_buf.episode_starts.get()[-1]
    assert not controller.was_reset


# --------------------------------------------------------------------------
# 8. Validation of malformed enabled configurations
# --------------------------------------------------------------------------


def _stds(obs_dim=4, pose_dim=3):
    return {
        "obs": np.ones(obs_dim, dtype=np.float64),
        "poses": np.ones(pose_dim, dtype=np.float64),
    }


@pytest.mark.parametrize(
    "cfg, exc",
    [
        ({"enabled": "true"}, TypeError),
        ({"enabled": 1}, TypeError),
        ({"enabled": True, "level": "0.1"}, TypeError),
        ({"enabled": True, "level": True}, TypeError),
        ({"enabled": True, "level": -0.1}, ValueError),
        ({"enabled": True, "level": float("nan")}, ValueError),
        ({"enabled": True, "level": float("inf")}, ValueError),
        ({"enabled": True, "seed": 1.5}, TypeError),
        ({"enabled": True, "seed": True}, TypeError),
        ({"enabled": True, "seed": "0"}, TypeError),
    ],
)
def test_invalid_enabled_config_raises(cfg, exc):
    with pytest.raises(exc):
        sn.make_sensor_noise_injector(cfg, _stds(), 4, 3, 0)


def test_non_mapping_config_raises():
    with pytest.raises(TypeError):
        sn.sensor_noise_enabled([("enabled", True)])


@pytest.mark.parametrize(
    "obs_std, pose_std, exc",
    [
        (np.ones((2, 2)), np.ones(3), ValueError),  # not 1-D
        (np.ones(5), np.ones(3), ValueError),  # obs dim mismatch
        (np.ones(4), np.ones(2), ValueError),  # pose dim mismatch
        (np.array([1.0, np.nan, 1.0, 1.0]), np.ones(3), ValueError),
        (np.array([1.0, np.inf, 1.0, 1.0]), np.ones(3), ValueError),
        (np.array([1.0, -1.0, 1.0, 1.0]), np.ones(3), ValueError),
        (None, np.ones(3), ValueError),  # missing scaler entry
        (np.ones(4), None, ValueError),
    ],
)
def test_invalid_scaler_stds_raise(obs_std, pose_std, exc):
    stds = {"obs": obs_std, "poses": pose_std}
    with pytest.raises(exc):
        sn.make_sensor_noise_injector({"enabled": True}, stds, 4, 3, 0)


def test_missing_scaler_entries_raise():
    class _NoScaler:
        scaler = None

    class _EmptyScaler:
        class scaler:
            scaler_params = {"acts": {"std": np.ones(3)}}

    with pytest.raises(ValueError):
        sn.scaler_stds_from_model(_NoScaler())
    with pytest.raises(ValueError):
        sn.scaler_stds_from_model(_EmptyScaler())


def test_invalid_yaw_index_raises():
    with pytest.raises(ValueError):
        sn.make_sensor_noise_injector({"enabled": True}, _stds(), 4, 3, 7)


def test_defaults_when_fields_are_omitted():
    injector = sn.make_sensor_noise_injector({"enabled": True}, _stds(), 4, 3, 0)
    assert injector.level == sn.DEFAULT_LEVEL
    assert injector.seed == sn.DEFAULT_SEED


def test_controller_construction_rejects_bad_config():
    with pytest.raises(ValueError):
        _make_controller(sensor_noise_cfg={"enabled": True, "level": -1.0})


# --------------------------------------------------------------------------
# Clean actual reward under sensor noise (MppiControllerWithAdaptation)
# --------------------------------------------------------------------------


def test_actual_reward_uses_clean_measurements():
    controller, config = _adapted_noisy_controller(seed=21)
    state0, pose0, act0 = _dummy_data(config)
    state1, pose1, act1 = _shifted_data(config, 1.3, 0.4)

    controller.initialize(state0, pose0, act0)
    controller.update(state1, pose1, act1)

    clean_prev = _clean_buf_item(config, state0, pose0, act0)
    clean_cur = _clean_buf_item(config, state1, pose1, act1)

    # 1. Both algorithmic buffers hold the same noisy item, which is not clean.
    ctrl_hist = controller.buf.get()
    adapt_hist = controller.adapt_buf.get()
    np.testing.assert_array_equal(ctrl_hist["obs"][-1], adapt_hist["obs"][-1])
    np.testing.assert_array_equal(ctrl_hist["pose"][-1], adapt_hist["pose"][-1])
    assert not np.allclose(ctrl_hist["obs"][-1], clean_cur["obs"])
    assert not np.allclose(ctrl_hist["pose"][-1], clean_cur["pose"])

    # The clean history holds exactly the unperturbed distilled measurements.
    clean_hist = controller.clean_reward_buf.get()
    assert len(clean_hist["obs"]) == 2
    np.testing.assert_array_equal(clean_hist["obs"][0], clean_prev["obs"])
    np.testing.assert_array_equal(clean_hist["pose"][0], clean_prev["pose"])
    np.testing.assert_array_equal(clean_hist["obs"][-1], clean_cur["obs"])
    np.testing.assert_array_equal(clean_hist["pose"][-1], clean_cur["pose"])

    # 2. The actual reward matches the independently assembled clean transition.
    actual = controller.compute_current_reward()
    expected_clean = _reward_from_items(controller, clean_prev, clean_cur)
    _assert_reward_equal(actual, expected_clean)

    # 3. ... and differs from the reward the noisy transition would give.
    expected_noisy = _reward_from_items(
        controller, _item_at(ctrl_hist, -2), _item_at(ctrl_hist, -1)
    )
    assert not np.isclose(actual["reward"], expected_noisy["reward"])


def test_planning_and_adaptation_inputs_stay_noisy():
    controller, config = _adapted_noisy_controller(seed=22)
    state0, pose0, act0 = _dummy_data(config)
    state1, pose1, act1 = _shifted_data(config, 1.3, 0.4)

    controller.initialize(state0, pose0, act0)
    controller.update(state1, pose1, act1)

    clean_cur = _clean_buf_item(config, state1, pose1, act1)
    noisy_hist = controller.buf.get()

    # 4. The dynamics-model inputs are taken from the noisy control history.
    world_model_inputs = controller._make_world_model_inputs()
    np.testing.assert_allclose(
        np.asarray(world_model_inputs["obs_hist"]),
        noisy_hist["obs"][-controller.obs_hist_len :],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(world_model_inputs["pose_hist"]),
        noisy_hist["pose"][-controller.pose_hist_len :],
        rtol=1e-6,
    )
    assert not np.allclose(np.asarray(world_model_inputs["obs_hist"])[-1], clean_cur["obs"])

    # The behavior-cloning inputs come from the same noisy buffer.
    bc_inputs = controller._make_model_inputs(controller._make_dummy_planner_poses())
    np.testing.assert_allclose(
        np.asarray(bc_inputs["obs_hist"])[0],
        noisy_hist["obs"][-controller.obs_hist_len :],
        rtol=1e-6,
    )

    # Adaptation batches are built from the noisy adaptation buffer.
    adapt_hist = controller.adapt_buf.get()
    assert not np.allclose(adapt_hist["obs"][-1], clean_cur["obs"])


def test_reset_starts_a_new_clean_reward_transition():
    controller, config = _adapted_noisy_controller(seed=23)
    state0, pose0, act0 = _dummy_data(config)
    state1, pose1, act1 = _shifted_data(config, 1.3, 0.4)
    state_r, pose_r, act_r = _shifted_data(config, 0.6, -0.7)
    state_n, pose_n, act_n = _shifted_data(config, 2.1, 0.9)

    controller.initialize(state0, pose0, act0)
    controller.update(state1, pose1, act1)

    # 5. After a reset the clean transition starts at the reset measurement.
    controller.reset(state_r, pose_r, act_r)
    clean_reset = _clean_buf_item(config, state_r, pose_r, act_r)
    clean_hist = controller.clean_reward_buf.get()
    for i in range(len(clean_hist["obs"])):
        np.testing.assert_array_equal(clean_hist["obs"][i], clean_reset["obs"])
        np.testing.assert_array_equal(clean_hist["pose"][i], clean_reset["pose"])

    controller.update(state_n, pose_n, act_n)
    clean_next = _clean_buf_item(config, state_n, pose_n, act_n)
    clean_hist = controller.clean_reward_buf.get()
    np.testing.assert_array_equal(clean_hist["obs"][0], clean_reset["obs"])
    np.testing.assert_array_equal(clean_hist["obs"][-1], clean_next["obs"])
    # Nothing from before the reset survives in the reward history.
    assert not np.allclose(clean_hist["obs"][0], clean_next["obs"])

    actual = controller.compute_current_reward()
    expected = _reward_from_items(controller, clean_reset, clean_next)
    _assert_reward_equal(actual, expected)


@pytest.mark.parametrize("sensor_noise_cfg", [None, {"enabled": False}])
def test_disabled_config_keeps_legacy_reward_path(sensor_noise_cfg):
    config_probe = _get_config("world_model")
    obs_dim, pose_dim, _ = _dims(config_probe)
    controller, config = _make_controller(
        sensor_noise_cfg=sensor_noise_cfg,
        obs_std=np.ones(obs_dim),
        pose_std=np.ones(pose_dim),
        cls=MppiControllerWithAdaptation,
        extra_args=(dict(ADAPTATION_CFG),),
    )
    state0, pose0, act0 = _dummy_data(config)
    state1, pose1, act1 = _shifted_data(config, 1.3, 0.4)

    controller.initialize(state0, pose0, act0)
    controller.update(state1, pose1, act1)

    # 6. No clean buffer is allocated and the reward still reads the main buffer.
    assert controller.clean_reward_buf is None
    hist = controller.buf.get()
    actual = controller.compute_current_reward()
    expected = _reward_from_items(controller, _item_at(hist, -2), _item_at(hist, -1))
    _assert_reward_equal(actual, expected)


def test_missing_clean_history_raises_instead_of_falling_back():
    controller, config = _adapted_noisy_controller(seed=24)
    state0, pose0, act0 = _dummy_data(config)
    state1, pose1, act1 = _shifted_data(config, 1.3, 0.4)
    controller.initialize(state0, pose0, act0)
    controller.update(state1, pose1, act1)

    # 7. Never silently fall back to the noisy main buffer.
    clean_buf = controller.clean_reward_buf
    controller.clean_reward_buf = None
    with pytest.raises(RuntimeError):
        controller.compute_current_reward()

    controller.clean_reward_buf = MultiRingBuffer(
        maxlen=2, example_item=_clean_buf_item(config, state0, pose0, act0)
    )
    with pytest.raises(RuntimeError):
        controller.compute_current_reward()

    controller.clean_reward_buf = clean_buf
    assert np.isfinite(controller.compute_current_reward()["reward"])


def test_clean_history_is_not_part_of_adaptation_state():
    controller, config = _adapted_noisy_controller(seed=25)
    state0, pose0, act0 = _dummy_data(config)
    state1, pose1, act1 = _shifted_data(config, 1.3, 0.4)
    controller.initialize(state0, pose0, act0)
    controller.update(state1, pose1, act1)

    # 8. The checkpointed adaptation buffer schema is unchanged and noisy.
    assert set(controller.adapt_buf.keys) == {"obs", "pose", "act", "path_line_def"}
    before = controller.clean_reward_buf.get()
    adapt_before = controller.adapt_buf.get()

    controller.save_adaptation()
    controller.load_adaptation()

    np.testing.assert_array_equal(
        controller.adapt_buf.get()["obs"], adapt_before["obs"]
    )
    np.testing.assert_array_equal(controller.clean_reward_buf.get()["obs"], before["obs"])
    clean_cur = _clean_buf_item(config, state1, pose1, act1)
    assert not np.allclose(controller.adapt_buf.get()["obs"][-1], clean_cur["obs"])

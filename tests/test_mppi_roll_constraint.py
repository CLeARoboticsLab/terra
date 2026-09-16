"""Focused tests for the optional MPPI roll-angle feasibility constraint.

The planner behavior is exercised through a stub controller that keeps the real
`_mppi_step`, `_update_action_distribution`, `_roll_feasible_mask`, and
roll-limit configuration code, but swaps the world model and reward function
for scripted ones. That makes predicted roll and candidate return directly
controllable, so each rule can be checked deterministically without a trained
model or a long control loop.
"""

import jax
import numpy as np
import jax.numpy as jnp
import pytest

import terra.utils.model as model_utils
from terra.control.mppi_controller import (
    DEFAULT_ROLL_LIMIT_DEG,
    ROLL_OBS_IDX,
    MppiController,
    MppiState,
)


OBS_DIM = 3
POSE_DIM = 4
ACT_DIM = 2
HORIZON = 3
NUM_SAMPLES = 8
NUM_ELITES = 4


class ScriptedWorldModel:
    """A world model whose predicted roll is a function of the actions.

    `roll_fn` maps a (candidate, T, act_dim) action batch to a (candidate, T)
    roll field in radians, exactly as a denormalized model output would be.
    """

    def __init__(self, roll_fn):
        self.roll_fn = roll_fn

    def inference(self, model_inputs, **kwargs):
        acts = model_inputs["fut_acts"]
        n_cand = acts.shape[0]
        roll = jnp.asarray(self.roll_fn(acts), dtype=jnp.float32)
        obs = jnp.zeros((n_cand, HORIZON, OBS_DIM), dtype=jnp.float32)
        obs = obs.at[..., ROLL_OBS_IDX].set(roll)
        return {
            "obs": obs,
            "poses": jnp.zeros((n_cand, HORIZON, POSE_DIM), dtype=jnp.float32),
        }


class StubPlanner(MppiController):
    """A `MppiController` reduced to what `_mppi_step` actually touches.

    The base `__init__` is deliberately not called: it would require a trained
    policy, a world model, and a full system config, none of which affect the
    code under test. Roll-limit configuration still goes through the real
    `_configure_roll_limit`, so validation and unit conversion are not
    reimplemented here.
    """

    def __init__(self, mppi_cfg, roll_fn, reward_fn):
        self.device = jax.devices()[0]
        self.T = HORIZON
        self.act_dim = ACT_DIM
        self.world_model = ScriptedWorldModel(roll_fn)
        self._scripted_reward_fn = reward_fn

        self.B = NUM_SAMPLES
        self.num_elites = NUM_ELITES
        self.num_bc_trajs = 2
        self.min_std = 0.001
        self.init_std = jnp.array(1.0)
        self.action_min = jnp.array(-1.0)
        self.action_max = jnp.array(1.0)
        self.bc_act_std = jnp.array(0.025)
        self.temperature = 0.25
        self.momentum = 0.0
        self.discount = 0.99
        self.J = mppi_cfg.get("iterations", 2)
        self.shift = 1
        self.discounts = self.discount ** jnp.arange(self.T)
        self._action_noise_fn = self._white_noise

        self._configure_roll_limit(mppi_cfg)

    def _calc_reward(self, path_def, model_inputs, model_outputs):
        rewards = jnp.asarray(
            self._scripted_reward_fn(model_inputs["fut_acts"], model_outputs),
            dtype=jnp.float32,
        )
        return rewards, {}, {}

    def make_state(self, seed=0):
        return MppiState(
            mean=self._to_dev_f32(jnp.zeros((self.T, self.act_dim))),
            std=self._to_dev_f32(jnp.ones((self.T, self.act_dim))),
            key_data=self._to_key_data(jax.random.PRNGKey(seed)),
        )

    def step(self, state=None):
        if state is None:
            state = self.make_state()
        return self._mppi_step(
            state,
            jnp.zeros((self.T, self.act_dim), dtype=jnp.float32),
            {
                "obs_hist": jnp.zeros((1, OBS_DIM), dtype=jnp.float32),
                "pose_hist": jnp.zeros((1, POSE_DIM), dtype=jnp.float32),
                "acts_hist": jnp.zeros((1, ACT_DIM), dtype=jnp.float32),
            },
            jnp.zeros((HORIZON, 4), dtype=jnp.float32),
        )


def flat_roll(value):
    """Roll field that ignores the actions and is `value` everywhere."""

    def roll_fn(acts):
        return jnp.full(acts.shape[:2], value, dtype=jnp.float32)

    return roll_fn


def sum_of_actions(acts, model_outputs):
    """A return that strictly increases with the magnitude of the actions."""
    return jnp.sum(acts, axis=(1, 2))


def make_planner(mppi_cfg=None, roll_fn=None, reward_fn=sum_of_actions):
    return StubPlanner(
        mppi_cfg if mppi_cfg is not None else {},
        roll_fn if roll_fn is not None else flat_roll(0.0),
        reward_fn,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_missing_config_leaves_constraint_disabled():
    planner = make_planner({})
    assert planner.roll_limit_enabled is False
    assert planner.roll_limit_rad is None


def test_explicit_disable_leaves_constraint_disabled():
    planner = make_planner({"roll_limit_enabled": False, "roll_limit_deg": 15.0})
    assert planner.roll_limit_enabled is False
    assert planner.roll_limit_rad is None


def test_supplying_a_limit_does_not_implicitly_enable():
    planner = make_planner({"roll_limit_deg": 5.0})
    assert planner.roll_limit_enabled is False
    assert planner.roll_limit_rad is None


def test_fifteen_degrees_becomes_pi_over_twelve_radians():
    planner = make_planner({"roll_limit_enabled": True, "roll_limit_deg": 15.0})
    assert planner.roll_limit_enabled is True
    assert planner.roll_limit_rad == pytest.approx(np.pi / 12)


def test_enabled_without_a_limit_uses_the_fifteen_degree_default():
    planner = make_planner({"roll_limit_enabled": True})
    assert planner.roll_limit_rad == pytest.approx(np.deg2rad(DEFAULT_ROLL_LIMIT_DEG))
    assert planner.roll_limit_rad == pytest.approx(np.pi / 12)


@pytest.mark.parametrize(
    "bad_limit",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "15.0", None],
)
def test_invalid_enabled_limit_raises(bad_limit):
    with pytest.raises(ValueError):
        make_planner({"roll_limit_enabled": True, "roll_limit_deg": bad_limit})


@pytest.mark.parametrize("bad_limit", [0.0, -1.0, float("nan"), "15.0", None])
def test_disabled_malformed_limit_is_inert(bad_limit):
    planner = make_planner({"roll_limit_enabled": False, "roll_limit_deg": bad_limit})
    assert planner.roll_limit_enabled is False
    assert planner.roll_limit_rad is None


@pytest.mark.parametrize("bad_flag", [1, 0, "true", "false", None])
def test_non_bool_enable_flag_raises(bad_flag):
    with pytest.raises(ValueError):
        make_planner({"roll_limit_enabled": bad_flag})


# ---------------------------------------------------------------------------
# Feasibility semantics
# ---------------------------------------------------------------------------


def mask_for(rolls, limit_deg=15.0):
    """Runs the real feasibility mask over a (candidate, time) roll field."""
    planner = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": limit_deg}
    )
    rolls = jnp.asarray(rolls, dtype=jnp.float32)
    obs = jnp.zeros(rolls.shape + (OBS_DIM,), dtype=jnp.float32)
    obs = obs.at[..., ROLL_OBS_IDX].set(rolls)
    return np.asarray(planner._roll_feasible_mask(obs))


def test_rolls_within_the_limit_are_feasible():
    limit = np.float32(np.deg2rad(15.0))
    mask = mask_for([[0.0, 0.1 * limit, 0.9 * limit], [-0.5 * limit, 0.0, 0.0]])
    assert mask.tolist() == [True, True]


def test_rolls_exactly_on_either_boundary_are_feasible():
    limit = np.float32(np.deg2rad(15.0))
    mask = mask_for([[limit, 0.0, 0.0], [-limit, 0.0, 0.0], [limit, -limit, limit]])
    assert mask.tolist() == [True, True, True]


def test_rolls_beyond_either_boundary_are_rejected():
    limit = np.float32(np.deg2rad(15.0))
    eps = np.float32(1e-4)
    mask = mask_for([[limit + eps, 0.0, 0.0], [-limit - eps, 0.0, 0.0]])
    assert mask.tolist() == [False, False]


def test_a_violation_at_any_future_step_rejects_the_candidate():
    limit = np.float32(np.deg2rad(15.0))
    over = 2.0 * limit
    mask = mask_for(
        [
            [over, 0.0, 0.0],  # first step
            [0.0, over, 0.0],  # middle step
            [0.0, 0.0, over],  # last step
            [0.0, 0.0, 0.0],  # none
        ]
    )
    assert mask.tolist() == [False, False, False, True]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_roll_rejects_the_candidate(bad):
    mask = mask_for([[0.0, bad, 0.0], [0.0, 0.0, 0.0]])
    assert mask.tolist() == [False, True]


def test_only_the_roll_channel_is_inspected():
    limit = np.float32(np.deg2rad(15.0))
    planner = make_planner({"roll_limit_enabled": True, "roll_limit_deg": 15.0})
    obs = jnp.full((2, HORIZON, OBS_DIM), 10.0 * limit, dtype=jnp.float32)
    obs = obs.at[..., ROLL_OBS_IDX].set(0.0)
    assert np.asarray(planner._roll_feasible_mask(obs)).tolist() == [True, True]


def test_current_observation_is_not_part_of_the_check():
    """A rover already past the limit can still plan a feasible recovery."""
    limit = float(np.deg2rad(15.0))
    planner = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": 15.0},
        roll_fn=flat_roll(0.0),
    )
    # The measured history is far outside the bound; predictions are inside it.
    out = planner._mppi_step(
        planner.make_state(),
        jnp.zeros((HORIZON, ACT_DIM), dtype=jnp.float32),
        {
            "obs_hist": jnp.full((1, OBS_DIM), 5.0 * limit, dtype=jnp.float32),
            "pose_hist": jnp.zeros((1, POSE_DIM), dtype=jnp.float32),
            "acts_hist": jnp.zeros((1, ACT_DIM), dtype=jnp.float32),
        },
        jnp.zeros((HORIZON, 4), dtype=jnp.float32),
    )
    unconstrained = make_planner({}, roll_fn=flat_roll(0.0)).step()
    assert np.allclose(np.asarray(out.mean), np.asarray(unconstrained.mean))
    assert np.all(np.isfinite(np.asarray(out.mean)))


# ---------------------------------------------------------------------------
# Planner behavior
# ---------------------------------------------------------------------------


def test_disabled_planning_matches_the_legacy_path_exactly():
    """Missing keys and explicit disablement both reproduce legacy MPPI output."""
    roll_fn = flat_roll(np.deg2rad(45.0))  # would be rejected if enabled
    legacy = make_planner({}, roll_fn=roll_fn).step()
    disabled = make_planner(
        {"roll_limit_enabled": False, "roll_limit_deg": 15.0}, roll_fn=roll_fn
    ).step()

    assert np.array_equal(np.asarray(legacy.mean), np.asarray(disabled.mean))
    assert np.array_equal(np.asarray(legacy.std), np.asarray(disabled.std))
    assert np.array_equal(
        np.asarray(legacy.key_data), np.asarray(disabled.key_data)
    )


def test_all_feasible_planning_matches_the_unconstrained_result():
    """With every candidate inside the bound the mask changes no selection."""
    roll_fn = flat_roll(np.deg2rad(1.0))
    unconstrained = make_planner({}, roll_fn=roll_fn).step()
    constrained = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": 15.0}, roll_fn=roll_fn
    ).step()

    assert np.allclose(
        np.asarray(unconstrained.mean), np.asarray(constrained.mean), atol=1e-6
    )
    assert np.allclose(
        np.asarray(unconstrained.std), np.asarray(constrained.std), atol=1e-6
    )


def test_high_return_invalid_candidate_cannot_influence_the_update():
    """The best-return candidate is infeasible; the plan must follow the rest.

    Roll grows with the first action channel, so pushing that channel up is both
    the highest-return move and the one the constraint forbids. Without masking
    the refit mean would climb into the forbidden region; with masking it must
    stay where the predicted roll is admissible.
    """
    limit_deg = 15.0
    limit = float(np.deg2rad(limit_deg))
    cfg = {"iterations": 8}

    def roll_fn(acts):
        # Roll is proportional to the first action channel: |a0| = 1 sits at
        # twice the limit, so only candidates with |a0| <= 0.5 are feasible.
        return 2.0 * limit * acts[..., 0]

    def reward_fn(acts, model_outputs):
        # Strictly increasing in the first action channel.
        return jnp.sum(acts[..., 0], axis=1)

    unconstrained = make_planner(cfg, roll_fn=roll_fn, reward_fn=reward_fn).step()
    constrained = make_planner(
        {**cfg, "roll_limit_enabled": True, "roll_limit_deg": limit_deg},
        roll_fn=roll_fn,
        reward_fn=reward_fn,
    ).step()

    unconstrained_a0 = np.asarray(unconstrained.mean)[:, 0]
    constrained_a0 = np.asarray(constrained.mean)[:, 0]

    # The unconstrained planner chases the forbidden high-return region.
    assert np.any(unconstrained_a0 > 0.5)
    # The constrained planner stays where the predicted roll is admissible, and
    # is held strictly back from the return-maximizing direction at every step.
    assert np.all(constrained_a0 <= 0.5 + 1e-5)
    assert np.all(constrained_a0 < unconstrained_a0)
    assert np.all(np.isfinite(constrained_a0))


def test_candidate_origin_does_not_matter():
    """BC proposals are masked by the same rule as random samples.

    The BC block is the tail of the concatenated batch. Making only that block
    infeasible must remove it from the update, which a shared mask does and a
    sampled-only mask would not.
    """
    limit_deg = 15.0
    limit = float(np.deg2rad(limit_deg))
    n_bc = 2

    def roll_fn(acts):
        # Only the trailing BC candidates violate the bound.
        n = acts.shape[0]
        idx = jnp.arange(n)[:, None]
        is_bc = idx >= (n - n_bc)
        return jnp.where(is_bc, 4.0 * limit, 0.0) * jnp.ones(
            (n, HORIZON), dtype=jnp.float32
        )

    def reward_fn(acts, model_outputs):
        # BC candidates would otherwise dominate the elite set.
        n = acts.shape[0]
        return jnp.where(jnp.arange(n) >= (n - n_bc), 100.0, jnp.sum(acts[..., 0], 1))

    planner = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": limit_deg},
        roll_fn=roll_fn,
        reward_fn=reward_fn,
    )
    constrained = planner.step()
    unconstrained = make_planner(
        {}, roll_fn=roll_fn, reward_fn=reward_fn
    ).step()

    assert np.all(np.isfinite(np.asarray(constrained.mean)))
    # The dominant BC block is excluded, so the plan differs from the
    # unconstrained one that is pulled entirely toward those candidates.
    assert not np.allclose(
        np.asarray(constrained.mean), np.asarray(unconstrained.mean), atol=1e-5
    )


def test_zero_feasible_iteration_preserves_the_incoming_distribution():
    """Every candidate violates the bound, so no iteration may update anything."""
    planner = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": 15.0, "iterations": 4},
        roll_fn=flat_roll(np.deg2rad(90.0)),
    )
    prev_mean = jnp.asarray(
        np.linspace(-0.4, 0.4, HORIZON * ACT_DIM, dtype=np.float32).reshape(
            HORIZON, ACT_DIM
        )
    )
    state = MppiState(
        mean=planner._to_dev_f32(prev_mean),
        std=planner._to_dev_f32(jnp.ones((HORIZON, ACT_DIM))),
        key_data=planner._to_key_data(jax.random.PRNGKey(0)),
    )
    out = planner.step(state)

    # The plan is exactly the shifted previous mean, with the initialized std.
    expected_mean = jnp.roll(prev_mean, shift=1, axis=0).at[1:].set(0.0)
    assert np.allclose(np.asarray(out.mean), np.asarray(expected_mean))
    assert np.allclose(np.asarray(out.std), np.ones((HORIZON, ACT_DIM)))
    assert np.all(np.isfinite(np.asarray(out.mean)))
    assert np.all(np.isfinite(np.asarray(out.std)))


def test_zero_feasible_iterations_produce_no_nans_for_any_horizon_of_iterations():
    for iterations in (1, 2, 6):
        planner = make_planner(
            {
                "roll_limit_enabled": True,
                "roll_limit_deg": 15.0,
                "iterations": iterations,
            },
            roll_fn=flat_roll(np.nan),
        )
        out = planner.step()
        assert np.all(np.isfinite(np.asarray(out.mean)))
        assert np.all(np.isfinite(np.asarray(out.std)))


def test_a_later_feasible_iteration_still_updates_after_an_infeasible_one():
    """Zero-feasible iterations are skipped, not fatal to the whole plan."""
    limit_deg = 15.0
    limit = float(np.deg2rad(limit_deg))

    def roll_fn(acts):
        # Feasible only where the sampled first channel happens to be small;
        # with std = 1 on the first iteration and a refit std afterwards, both
        # branches of the conditional are exercised across iterations.
        return 4.0 * limit * acts[..., 0]

    planner = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": limit_deg, "iterations": 6},
        roll_fn=roll_fn,
    )
    out = planner.step()
    assert np.all(np.isfinite(np.asarray(out.mean)))
    assert np.all(np.isfinite(np.asarray(out.std)))
    assert np.all(np.abs(np.asarray(out.mean)[:, 0]) <= 0.25 + 1e-5)


# ---------------------------------------------------------------------------
# Reward reporting is untouched
# ---------------------------------------------------------------------------


def test_reward_calculation_is_never_masked():
    """`_calc_reward` sees no `-inf` even when every candidate is infeasible."""
    seen = []

    def reward_fn(acts, model_outputs):
        rewards = jnp.sum(acts[..., 0], axis=1)
        jax.debug.callback(lambda r: seen.append(np.asarray(r)), rewards)
        return rewards

    planner = make_planner(
        {"roll_limit_enabled": True, "roll_limit_deg": 15.0, "iterations": 2},
        roll_fn=flat_roll(np.deg2rad(90.0)),
        reward_fn=reward_fn,
    )
    planner.step()

    assert seen, "the reward function was never called"
    for rewards in seen:
        assert np.all(np.isfinite(rewards))


def make_real_controller(mppi_cfg):
    """Builds a genuine `MppiController` over the shared dummy test fixtures."""
    from test_control import _dummy_data, _get_config, _make_dummy_model
    from terra.control.rewards import ErnestRewardFunction

    bc_cfg = _get_config("behavior_cloning_with_pose_hist")
    world_model_cfg = _get_config("world_model")
    state, pose, act = _dummy_data(bc_cfg)
    controller = MppiController(
        _make_dummy_model(bc_cfg),
        bc_cfg,
        _make_dummy_model(world_model_cfg),
        world_model_cfg,
        dict(mppi_cfg),
        ErnestRewardFunction({}),
    )
    controller.initialize(state, pose, act)
    controller.update(state, pose, act)
    return controller, (state, pose, act)


def assert_all_finite(reward_data):
    assert np.isfinite(reward_data["reward"]), reward_data["reward"]
    for key in ("reward_comps", "reward_comps_unweighted"):
        for name, value in reward_data[key].items():
            assert np.all(np.isfinite(value)), f"{key}.{name} = {value}"


def test_public_reward_reporting_is_unaffected_by_the_constraint():
    """`compute_current_reward` and selected-trajectory reporting stay ordinary.

    This runs the real controller rather than the stub, so `_calc_reward`,
    `compute_current_reward`, and `_predict_future_trajectory` are the
    production implementations. The enabled controller uses a limit far tighter
    than the untrained dummy model can satisfy, which forces the all-infeasible
    fallback: exactly the case where a leaked `-inf` would surface in a reported
    reward.
    """
    base = {"num_samples": 64, "num_elites": 8, "iterations": 2}
    disabled, _ = make_real_controller(base)
    enabled, _ = make_real_controller(
        {**base, "roll_limit_enabled": True, "roll_limit_deg": 1e-6}
    )

    # The constraint never reaches the measured-transition reward, which depends
    # only on the buffers, so both controllers must agree exactly.
    disabled_now = disabled.compute_current_reward()
    enabled_now = enabled.compute_current_reward()
    assert_all_finite(disabled_now)
    assert_all_finite(enabled_now)
    assert enabled_now["reward"] == disabled_now["reward"]
    assert enabled_now["reward_comps"] == disabled_now["reward_comps"]
    assert (
        enabled_now["reward_comps_unweighted"]
        == disabled_now["reward_comps_unweighted"]
    )

    # Selected-trajectory reporting stays a finite evaluation quantity even when
    # the plan is the untouched fallback from an all-infeasible search.
    disabled_out = disabled.run_control(disabled._make_dummy_planner_poses())
    enabled_out = enabled.run_control(enabled._make_dummy_planner_poses())
    assert_all_finite(disabled_out)
    assert_all_finite(enabled_out)
    assert np.all(np.isfinite(enabled_out["actions"]))
    assert np.all(np.isfinite(enabled_out["std"]))
    assert np.all(np.isfinite(enabled_out["predicted_poses"]))

    # The reported components are the same named quantities in both modes; the
    # constraint adds, removes, and renames nothing.
    assert enabled_out["reward_comps"].keys() == disabled_out["reward_comps"].keys()
    assert (
        enabled_out["reward_comps_unweighted"].keys()
        == disabled_out["reward_comps_unweighted"].keys()
    )

    # The limit really was unsatisfiable, so the fallback path was exercised.
    model_inputs = enabled._make_world_model_inputs()
    batched = model_utils.repeat_along_batch_dim(model_inputs, 4)
    batched["fut_acts"] = jnp.zeros((4, enabled.T, enabled.act_dim))
    pred_obs = enabled.world_model.inference(batched)["obs"]
    assert pred_obs.ndim == 3, "predicted obs must be (candidate, time, obs_dim)"
    assert not np.any(np.asarray(enabled._roll_feasible_mask(pred_obs)))

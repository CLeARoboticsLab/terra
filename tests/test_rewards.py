"""Focused tests for the soft roll-limit Huber reward component.

The component penalizes `next_obs[..., 0]` (roll, in radians) whenever its
magnitude exceeds a configurable limit, using a Huber on the normalized
exceedance. It defaults to zero weight so the existing scalar reward and
planning behavior are unchanged unless explicitly enabled.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from terra.control.rewards import ErnestRewardFunction


OBS_DIM = ErnestRewardFunction.WHEEL_FORCE_START_IDX + ErnestRewardFunction.WHEEL_FORCE_DIM
POSE_DIM = 5
ACT_DIM = 3
PATH_DIM = 4

DELTA = ErnestRewardFunction.ROLL_LIMIT_HUBER_DELTA
LIMIT_DEG = ErnestRewardFunction.DEFAULT_ROLL_LIMIT_DEG

# (absolute roll in degrees, unweighted roll_limit component)
ANALYTICAL_TABLE = [
    (14.0, 0.000000),
    (15.0, 0.000000),
    (16.0, -0.002222),
    (17.0, -0.008889),
    (18.0, -0.020000),
    (19.0, -0.035417),
    (20.0, -0.052083),
    (21.0, -0.068750),
    (22.0, -0.085417),
    (23.0, -0.102083),
    (24.0, -0.118750),
    (25.0, -0.135417),
]


def make_inputs(roll_rad, batch=()):
    """Build reward inputs whose only nonzero content is the obs roll channel."""
    roll = jnp.asarray(roll_rad, dtype=jnp.float32)
    shape = batch + roll.shape if roll.ndim else batch
    path_def = jnp.zeros(shape + (PATH_DIM,), dtype=jnp.float32)
    # b = 1 so forward advance is a plain delta-x; a = c = 0.
    path_def = path_def.at[..., ErnestRewardFunction.B_IDX].set(1.0)
    pose = jnp.zeros(shape + (POSE_DIM,), dtype=jnp.float32)
    action = jnp.zeros(shape + (ACT_DIM,), dtype=jnp.float32)
    next_pose = jnp.zeros(shape + (POSE_DIM,), dtype=jnp.float32)
    next_obs = jnp.zeros(shape + (OBS_DIM,), dtype=jnp.float32)
    next_obs = next_obs.at[..., ErnestRewardFunction.OBS_ROLL_IDX].set(roll)
    return path_def, pose, action, next_pose, next_obs


def call(cfg, roll_rad, batch=()):
    fn = ErnestRewardFunction(cfg)
    return fn(*make_inputs(roll_rad, batch=batch))


def test_defaults_when_config_omits_keys():
    fn = ErnestRewardFunction({})
    assert fn.weight_roll_limit == 0.0
    assert fn.roll_limit_deg == 15.0
    assert fn.roll_limit_rad == pytest.approx(np.deg2rad(15.0))


def test_zero_weight_leaves_total_reward_unchanged_but_reports_unweighted():
    roll = float(np.deg2rad(25.0))
    reward, comps, unweighted = call({}, roll)

    # Total reward equals the sum of the pre-existing weighted components.
    legacy_total = sum(v for k, v in comps.items() if k != "roll_limit")
    assert float(reward) == pytest.approx(float(legacy_total), abs=1e-6)

    assert float(comps["roll_limit"]) == 0.0
    assert float(unweighted["roll_limit"]) == pytest.approx(-0.135417, abs=1e-5)


def test_component_keys_are_stable_across_weights():
    expected = {
        "forward_advance",
        "lateral_deviation",
        "bearing_deviation",
        "yaw",
        "roll",
        "yaw_action",
        "roll_action",
        "wheel_forces",
        "roll_limit",
    }
    for weight in (0.0, 100.0):
        _, comps, unweighted = call({"weight_roll_limit": weight}, 0.0)
        assert set(comps) == expected
        assert set(unweighted) == expected


def test_existing_components_unaffected_by_the_new_term():
    roll = float(np.deg2rad(25.0))
    _, base_comps, base_unweighted = call({}, roll)
    _, comps, unweighted = call({"weight_roll_limit": 100.0}, roll)
    for key in base_comps:
        if key == "roll_limit":
            continue
        assert float(comps[key]) == pytest.approx(float(base_comps[key]))
        assert float(unweighted[key]) == pytest.approx(float(base_unweighted[key]))


@pytest.mark.parametrize("deg,expected", ANALYTICAL_TABLE)
def test_analytical_values_at_weight_100(deg, expected):
    roll = float(np.deg2rad(deg))
    _, comps, unweighted = call({"weight_roll_limit": 100.0}, roll)
    assert float(unweighted["roll_limit"]) == pytest.approx(expected, abs=1e-5)
    assert float(comps["roll_limit"]) == pytest.approx(100.0 * expected, abs=1e-3)


@pytest.mark.parametrize("deg,expected", ANALYTICAL_TABLE)
def test_penalty_is_symmetric_in_roll_sign(deg, expected):
    cfg = {"weight_roll_limit": 100.0}
    _, _, pos = call(cfg, float(np.deg2rad(deg)))
    _, _, neg = call(cfg, float(-np.deg2rad(deg)))
    assert float(pos["roll_limit"]) == pytest.approx(float(neg["roll_limit"]))
    assert float(pos["roll_limit"]) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("deg", [0.0, 5.0, 14.999, 15.0, -15.0, -5.0])
def test_no_penalty_at_or_below_the_limit(deg):
    _, comps, unweighted = call({"weight_roll_limit": 100.0}, float(np.deg2rad(deg)))
    assert float(unweighted["roll_limit"]) == pytest.approx(0.0, abs=1e-9)
    assert float(comps["roll_limit"]) == pytest.approx(0.0, abs=1e-6)


def test_quadratic_to_linear_boundary_is_continuous():
    # excess == delta == 0.25 at 18.75 degrees.
    boundary_deg = LIMIT_DEG * (1.0 + DELTA)
    assert boundary_deg == pytest.approx(18.75)

    def unweighted_at(deg):
        _, _, unweighted = call({}, float(np.deg2rad(deg)))
        return float(unweighted["roll_limit"])

    eps = 1e-3
    below, at, above = (
        unweighted_at(boundary_deg - eps),
        unweighted_at(boundary_deg),
        unweighted_at(boundary_deg + eps),
    )
    assert at == pytest.approx(-0.5 * DELTA**2, abs=1e-6)
    assert at == pytest.approx(-DELTA * (DELTA - 0.5 * DELTA), abs=1e-6)
    assert below == pytest.approx(at, abs=1e-4)
    assert above == pytest.approx(at, abs=1e-4)


def roll_limit_obs_grad(cfg, deg):
    """d(total reward)/d(next_obs) at a given roll, in radians^-1."""
    fn = ErnestRewardFunction(cfg)
    path_def, pose, action, next_pose, next_obs = make_inputs(float(np.deg2rad(deg)))

    def total(obs):
        return fn(path_def, pose, action, next_pose, obs)[0]

    return jax.grad(total)(next_obs)


def test_gradient_is_zero_inside_the_limit():
    grad = roll_limit_obs_grad({"weight_roll_limit": 100.0}, 10.0)
    assert float(grad[ErnestRewardFunction.OBS_ROLL_IDX]) == pytest.approx(0.0, abs=1e-6)


def test_gradient_grows_linearly_in_the_quadratic_region():
    cfg = {"weight_roll_limit": 100.0}
    limit_rad = float(np.deg2rad(LIMIT_DEG))
    for deg in (16.0, 17.0, 18.0):
        excess = np.deg2rad(deg) / limit_rad - 1.0
        expected = -100.0 * excess / limit_rad
        got = float(roll_limit_obs_grad(cfg, deg)[ErnestRewardFunction.OBS_ROLL_IDX])
        assert got == pytest.approx(expected, rel=1e-4)


def test_gradient_is_capped_in_the_linear_region_and_sign_correct():
    cfg = {"weight_roll_limit": 100.0}
    limit_rad = float(np.deg2rad(LIMIT_DEG))
    cap = 100.0 * DELTA / limit_rad
    assert cap == pytest.approx(95.49, abs=0.01)

    for deg in (20.0, 25.0, 40.0):
        pos = float(roll_limit_obs_grad(cfg, deg)[ErnestRewardFunction.OBS_ROLL_IDX])
        neg = float(roll_limit_obs_grad(cfg, -deg)[ErnestRewardFunction.OBS_ROLL_IDX])
        # Reward decreases as |roll| grows, so the gradient opposes the roll sign.
        assert pos == pytest.approx(-cap, rel=1e-4)
        assert neg == pytest.approx(cap, rel=1e-4)


def test_only_the_roll_channel_receives_roll_limit_gradient():
    cfg = {"weight_roll_limit": 100.0}
    with_weight = roll_limit_obs_grad(cfg, 25.0)
    without_weight = roll_limit_obs_grad({}, 25.0)
    diff = np.asarray(with_weight - without_weight)
    assert diff[ErnestRewardFunction.OBS_ROLL_IDX] != 0.0
    assert np.allclose(diff[ErnestRewardFunction.OBS_ROLL_IDX + 1 :], 0.0)


@pytest.mark.parametrize("bad_roll", [np.nan, np.inf, -np.inf])
def test_non_finite_roll_yields_the_fixed_finite_penalty(bad_roll):
    _, comps, unweighted = call({"weight_roll_limit": 100.0}, float(bad_roll))
    assert float(unweighted["roll_limit"]) == pytest.approx(-0.21875, abs=1e-6)
    assert float(comps["roll_limit"]) == pytest.approx(-21.875, abs=1e-3)


@pytest.mark.parametrize("bad_roll", [np.nan, np.inf, -np.inf])
def test_zero_weight_never_produces_nan_from_non_finite_roll(bad_roll):
    _, comps, unweighted = call({}, float(bad_roll))
    assert float(comps["roll_limit"]) == 0.0
    assert np.isfinite(float(unweighted["roll_limit"]))


@pytest.mark.parametrize("bad_roll", [np.nan, np.inf, -np.inf])
def test_non_finite_roll_does_not_leak_into_gradients(bad_roll):
    fn = ErnestRewardFunction({"weight_roll_limit": 100.0})
    path_def, pose, action, next_pose, next_obs = make_inputs(float(bad_roll))

    def roll_limit_only(obs):
        return fn(path_def, pose, action, next_pose, obs)[2]["roll_limit"]

    grad = jax.grad(roll_limit_only)(next_obs)
    assert np.isfinite(np.asarray(grad)).all()
    assert float(grad[ErnestRewardFunction.OBS_ROLL_IDX]) == 0.0


def test_batched_rolls_are_handled_elementwise():
    degs = np.array([0.0, 16.0, -20.0, 25.0])
    _, _, unweighted = call({"weight_roll_limit": 100.0}, np.deg2rad(degs))
    expected = [0.0, -0.002222, -0.052083, -0.135417]
    np.testing.assert_allclose(
        np.asarray(unweighted["roll_limit"]), expected, atol=1e-5
    )


def test_custom_limit_rescales_the_penalty():
    _, _, unweighted = call(
        {"weight_roll_limit": 1.0, "roll_limit_deg": 30.0}, float(np.deg2rad(32.0))
    )
    excess = 2.0 / 30.0
    assert float(unweighted["roll_limit"]) == pytest.approx(-0.5 * excess**2, abs=1e-7)


@pytest.mark.parametrize(
    "value", [-1.0, np.nan, np.inf, -np.inf, "0.0", True, False, None]
)
def test_invalid_weight_rejected(value):
    with pytest.raises(ValueError):
        ErnestRewardFunction({"weight_roll_limit": value})


@pytest.mark.parametrize(
    "value", [0.0, -1.0, np.nan, np.inf, -np.inf, "15.0", True, False, None]
)
def test_invalid_limit_rejected(value):
    with pytest.raises(ValueError):
        ErnestRewardFunction({"roll_limit_deg": value})


@pytest.mark.parametrize("value", [0, 0.0, 100, 100.0, np.float32(2.5), np.int64(3)])
def test_valid_real_scalar_weights_accepted(value):
    fn = ErnestRewardFunction({"weight_roll_limit": value})
    assert fn.weight_roll_limit == pytest.approx(float(value))


@pytest.mark.parametrize("value", [1, 15.0, np.float64(30.0), np.int32(20)])
def test_valid_real_scalar_limits_accepted(value):
    fn = ErnestRewardFunction({"roll_limit_deg": value})
    assert fn.roll_limit_deg == pytest.approx(float(value))


def test_adaptation_style_reward_scaled_emphasis():
    """Mirror `AdaptationWrapper`: grad of reward wrt next_obs scales obs error."""
    roll = float(np.deg2rad(25.0))
    path_def, pose, action, next_pose, next_obs = make_inputs(roll)
    obs_error = jnp.full_like(next_obs, 0.1)

    def emphasis(cfg):
        fn = ErnestRewardFunction(cfg)

        def reward_only(next_pose_, next_obs_):
            return fn(path_def, pose, action, next_pose_, next_obs_)[0]

        grad_pose, grad_obs = jax.grad(reward_only, argnums=[0, 1])(next_pose, next_obs)
        del grad_pose
        per_dim = (grad_obs * obs_error) ** 2
        return float(per_dim[ErnestRewardFunction.OBS_ROLL_IDX]), float(
            jnp.sum(per_dim)
        )

    off_roll, off_total = emphasis({})
    on_roll, on_total = emphasis({"weight_roll_limit": 100.0})

    assert off_roll == pytest.approx(0.0, abs=1e-9)
    assert on_roll > 1.0
    assert on_total > off_total

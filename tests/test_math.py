import terra.utils.math as math_utils
import terra.utils.model as model_utils
import numpy as np
import jax.numpy as jnp
import jax
import pytest


@pytest.fixture
def se2_idxs():
    return {"x": 0, "y": 1, "yaw": 2}


def test_identity_transform(se2_idxs):
    """Transforming to and from the same pose should result in no change."""
    poses = jnp.array([[[1.0, 2.0, 0.5], [1.5, 2.5, 0.6], [2.0, 3.0, 0.7]]])
    p0 = poses[:, 0, :]  # use first pose as reference

    body_frame = math_utils.express_in_body_frame(poses, p0, se2_idxs)
    world_frame = math_utils.express_in_world_frame(body_frame, p0, se2_idxs)

    assert jnp.allclose(world_frame, poses, atol=1e-6)


def test_zero_relative_pose(se2_idxs):
    """If the pose sequence equals p0, all outputs in body frame should be zeros."""
    T = 4
    p0 = jnp.array([[1.0, 2.0, 0.5], [3.0, 4.0, -0.2]])
    poses = jnp.tile(p0[:, None, :], (1, T, 1))

    body_frame = math_utils.express_in_body_frame(poses, p0, se2_idxs)

    expected = jnp.zeros_like(poses)
    assert jnp.allclose(body_frame, expected, atol=1e-6)


def test_translation_only(se2_idxs):
    """Translation with zero yaw should yield negative relative translation in
    body frame."""
    p0 = jnp.array([[1.0, 2.0, 0.0]])  # zero yaw
    pose = jnp.array([[[2.0, 4.0, 0.0]]])  # +1 x, +2 y in world frame

    body_frame = math_utils.express_in_body_frame(pose, p0, se2_idxs)
    expected = jnp.array([[[1.0, 2.0, 0.0]]])

    assert jnp.allclose(body_frame, expected, atol=1e-6)


def test_rotation_only(se2_idxs):
    """A point at (1,0) in world frame rotated -90 deg should land at (0,-1) in
    body frame."""
    p0 = jnp.array([[0.0, 0.0, jnp.pi / 2]])  # 90 deg
    pose = jnp.array([[[1.0, 0.0, 0.0]]])  # (1,0) in world

    body_frame = math_utils.express_in_body_frame(pose, p0, se2_idxs)
    expected_xy = jnp.array([[[0.0, -1.0]]])  # now at (0, -1)
    assert jnp.allclose(body_frame[:, :, :2], expected_xy, atol=1e-6)


def test_translation_only_world(se2_idxs):
    """A local (body) pose at (1,2) with p0 at origin and zero yaw should be
    (1,2) in world frame."""
    p0 = jnp.array([[0.0, 0.0, 0.0]])  # zero yaw
    body_pose = jnp.array([[[1.0, 2.0, 0.0]]])  # body frame

    world_frame = math_utils.express_in_world_frame(body_pose, p0, se2_idxs)
    expected = jnp.array([[[1.0, 2.0, 0.0]]])  # unchanged

    assert jnp.allclose(world_frame, expected, atol=1e-6)


def test_rotation_only_world(se2_idxs):
    """A local point at (0,-1) with p0 yaw=pi/2 should become (1, 0) in world frame."""
    p0 = jnp.array([[0.0, 0.0, jnp.pi / 2]])  # 90 degrees
    body_pose = jnp.array([[[0.0, -1.0, 0.0]]])  # in body frame, behind robot

    world_frame = math_utils.express_in_world_frame(body_pose, p0, se2_idxs)
    expected_xy = jnp.array([[[1.0, 0.0]]])  # rotated to world (1,0)

    assert jnp.allclose(world_frame[:, :, :2], expected_xy, atol=1e-6)


def test_round_trip_random(se2_idxs):
    """Round trip for random poses should recover original."""
    key = jax.random.PRNGKey(0)
    B, T, D = 4, 5, 3
    poses = jax.random.uniform(key, shape=(B, T, D), minval=-jnp.pi, maxval=jnp.pi)
    p0 = poses[:, 0, :]

    body_frame = math_utils.express_in_body_frame(poses, p0, se2_idxs)
    world_frame = math_utils.express_in_world_frame(body_frame, p0, se2_idxs)

    # allow small error due to numerical wrapping
    print("Original poses:", poses)
    print("World frame after round trip:", world_frame)
    assert jnp.allclose(world_frame, poses, atol=1e-5)


def test_merge_mean_var():
    data = np.random.normal(loc=1.0, scale=2.0, size=(1000, 5))
    block_size = 100

    mean_total = 0.0
    var_total = 0.0
    n_total = 0
    for i in range(0, len(data), block_size):
        block = data[i : i + block_size]
        mean_block = np.mean(block, axis=0)
        var_block = np.var(block, axis=0)

        mean_total, var_total, n_total = math_utils.merge_mean_var(
            mean_total, var_total, n_total, mean_block, var_block, len(block)
        )

    assert np.allclose(mean_total, np.mean(data, axis=0))
    assert np.allclose(var_total, np.var(data, axis=0))


def test_poses_scaler_params(se2_idxs):
    eps = 100
    ep_len = 1000
    H = 5
    T = 5
    key = jax.random.key(42)

    xs = jnp.arange(ep_len)
    ys = jnp.arange(ep_len) * 2
    yaws = jax.random.uniform(key, (ep_len,), minval=-1e-8, maxval=1e-8)
    qyaws = yaws
    qrolls = yaws

    poses = jnp.stack([xs, ys, yaws, qyaws, qrolls], axis=-1)
    poses = model_utils.repeat_along_batch_dim(poses, eps)

    psp = math_utils.poses_scaler_params(poses, se2_idxs, H, T)

    assert jnp.allclose(psp["mean"], jnp.array([2.0, 4.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert jnp.allclose(
        psp["std"], jnp.sqrt(jnp.array([2.0, 8.0, 0.0, 0.0, 0.0])), atol=1e-6
    )


def test_poses_scaler_params_variable_lengths(se2_idxs):
    H = 5
    T = 5

    # Create 2 episodes of different lengths
    xs1 = jnp.arange(10000)
    ys1 = xs1
    yaws1 = jnp.ones_like(xs1) * jnp.pi / 2
    qyaws1 = jnp.zeros_like(xs1)
    qrolls1 = jnp.zeros_like(xs1)
    poses1 = jnp.stack([xs1, ys1, yaws1, qyaws1, qrolls1], axis=-1)

    xs2 = jnp.arange(5000)
    ys2 = xs2 * 2
    yaws2 = jnp.zeros_like(xs2)
    qyaws2 = yaws2
    qrolls2 = yaws2
    poses2 = jnp.stack([xs2, ys2, yaws2, qyaws2, qrolls2], axis=-1)

    poses = [poses1, poses2]  # variable-length episodes

    # Run the scaler
    psp = math_utils.poses_scaler_params(poses, se2_idxs, H, T)

    # Verify
    expected_mean = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0])
    expected_std = jnp.sqrt(jnp.array([2.0, 12.0, 0.0, 0.0, 0.0]))

    assert jnp.allclose(
        psp["mean"], expected_mean, atol=1e-3
    ), f"Mean mismatch: {psp['mean']}"
    assert jnp.allclose(
        psp["std"], expected_std, atol=1e-3
    ), f"Std mismatch: {psp['std']}"

from terra.modeling import types
import jax.numpy as jnp
import jax
import tqdm


@jax.jit
def express_in_body_frame(
    pose_seq: jnp.ndarray, p0: jnp.ndarray, se2_idxs: dict[str, int]
):
    """Transforms all of the poses in pose_seq to the body frame with respect to p0.

    Arguments:
        pose_seq: sequence of poses, shape (B, T, pose_dim)
        p0: reference pose, shape (B, pose_dim)
        se2_idxs: dict with keys 'x', 'y', 'yaw' and values of indexes

    Returns:
    Transformed poses in the body frame, shape (B, T, pose_dim)
    """
    x_idx = se2_idxs["x"]
    y_idx = se2_idxs["y"]
    yaw_idx = se2_idxs["yaw"]

    dx = pose_seq[:, :, x_idx] - p0[:, None, x_idx]
    dy = pose_seq[:, :, y_idx] - p0[:, None, y_idx]
    dyaw = wrap_angle(
        wrap_angle(pose_seq[:, :, yaw_idx]) - wrap_angle(p0[:, None, yaw_idx])
    )

    # inverse rotation (world -> body) is equivalent to a rotation by -yaw
    cos_yaw = jnp.cos(-p0[:, yaw_idx])[:, None]
    sin_yaw = jnp.sin(-p0[:, yaw_idx])[:, None]

    x_body = dx * cos_yaw - dy * sin_yaw
    y_body = dx * sin_yaw + dy * cos_yaw

    pose_seq_body = pose_seq.at[:, :, x_idx].set(x_body)
    pose_seq_body = pose_seq_body.at[:, :, y_idx].set(y_body)
    pose_seq_body = pose_seq_body.at[:, :, yaw_idx].set(dyaw)

    return pose_seq_body


@jax.jit
def express_in_world_frame(
    pose_seq: jnp.ndarray, p0: jnp.ndarray, se2_idxs: dict[str, int]
):
    """Transforms all of the poses in pose_seq to the world frame with respect to p0.

    Arguments:
        pose_seq: sequence of poses in body frame, shape (B, T, pose_dim)
        p0: reference pose, shape (B, pose_dim)
        se2_idxs: dict with keys 'x', 'y', 'yaw' and values of indexes

    Returns:
    Transformed poses in the world frame, shape (B, T, pose_dim)
    """
    x_idx = se2_idxs["x"]
    y_idx = se2_idxs["y"]
    yaw_idx = se2_idxs["yaw"]

    x_body = pose_seq[:, :, x_idx]
    y_body = pose_seq[:, :, y_idx]
    dyaw = pose_seq[:, :, yaw_idx]

    cos_yaw = jnp.cos(p0[:, yaw_idx])[:, None]
    sin_yaw = jnp.sin(p0[:, yaw_idx])[:, None]

    dx = x_body * cos_yaw - y_body * sin_yaw
    dy = x_body * sin_yaw + y_body * cos_yaw

    x_world = dx + p0[:, None, x_idx]
    y_world = dy + p0[:, None, y_idx]
    yaw_world = wrap_angle(dyaw + p0[:, None, yaw_idx])

    pose_seq_world = pose_seq.at[:, :, x_idx].set(x_world)
    pose_seq_world = pose_seq_world.at[:, :, y_idx].set(y_world)
    pose_seq_world = pose_seq_world.at[:, :, yaw_idx].set(yaw_world)

    return pose_seq_world


def poses_scaler_params(
    poses: jnp.ndarray, se2_idxs: dict[str, int], H: int, T: int
) -> types.ScalerParam:
    """
    Computes the scaler params for episodes of poses. Supports both fixed-length
    JAX arrays and variable-length lists of JAX arrays.

    Arguments:
        poses: either a JAX array of shape (num_eps, ep_len, pose_dim),
               or a list of JAX arrays each of shape (ep_len_i, pose_dim)
        se2_idxs: dict with keys 'x', 'y', 'yaw' and values of indexes
        H: history length
        T: prediction length
    """
    if isinstance(poses, jnp.ndarray):
        return _poses_scaler_params(poses, se2_idxs, H, T)
    elif isinstance(poses, list):
        return _poses_scaler_params_variable(poses, se2_idxs, H, T)
    else:
        raise TypeError("poses must be a JAX array or a list of JAX arrays.")


def _poses_scaler_params(
    poses: jnp.ndarray, se2_idxs: dict[str, int], H: int, T: int
) -> types.ScalerParam:
    """
    Computes the scaler params for episodes of poses. For each episode, an
    averaging window of size max(H, T) is passed over the episode. The poses
    are converted to the body frame w.r.t. to the first pose in the window.

    Arguments:
        poses: pose data, shape (num_eps, ep_len, pose_dim)
        se2_idxs: dict with keys 'x', 'y', 'yaw' and values of indexes
        H: history length
        T: prediction length
    """
    mean = jnp.mean(poses, axis=(0, 1))
    var = jnp.var(poses, axis=(0, 1))

    block_size = max(H, T, 2)
    ep_len = poses.shape[1]
    idxs = jnp.array([se2_idxs["x"], se2_idxs["y"], se2_idxs["yaw"]])

    mean_total = 0.0
    var_total = 0.0
    n_total = 0
    for i in range(ep_len - block_size):
        block = poses[:, i : i + block_size, :]
        block = express_in_body_frame(block, block[:, 0, :], se2_idxs)
        mean_block = jnp.mean(block, axis=(0, 1))
        var_block = jnp.var(block, axis=(0, 1))
        mean_total, var_total, n_total = merge_mean_var(
            mean_total, var_total, n_total, mean_block, var_block, len(block)
        )

    mean = mean.at[idxs].set(mean_total[idxs])
    var = var.at[idxs].set(var_total[idxs])

    return {
        "mean": mean,
        "std": jnp.sqrt(var),
    }


def _poses_scaler_params_variable(
    poses: jnp.ndarray | list[jnp.ndarray],
    se2_idxs: dict[str, int],
    H: int,
    T: int,
) -> types.ScalerParam:
    """
    Computes the scaler params for episodes of poses. Supports a list of variable-length
    JAX arrays, each of shape (ep_len_i, pose_dim).

    Arguments:
        poses: either a JAX array of shape (num_eps, ep_len, pose_dim),
               or a list of JAX arrays each of shape (ep_len_i, pose_dim)
        se2_idxs: dict with keys 'x', 'y', 'yaw' and values of indexes
        H: history length
        T: prediction length
    """
    block_size = max(H, T, 2)
    idxs = jnp.array([se2_idxs["x"], se2_idxs["y"], se2_idxs["yaw"]])

    episode_list = poses

    # Gather all valid blocks into a list
    blocks = []
    for ep in tqdm.tqdm(episode_list, desc="Processing episodes for scaler params"):
        ep_len = ep.shape[0]
        if ep_len < block_size:
            continue
        # Stride across the episode
        for i in range(ep_len - block_size):
            block = ep[i : i + block_size, :]
            blocks.append(block)

    if len(blocks) == 0:
        raise ValueError("No valid blocks found. Check episode lengths and H/T.")

    # Stack all blocks (N, block_size, pose_dim)
    blocks = jnp.stack(blocks, axis=0)
    p0s = blocks[:, 0, :]  # (N, pose_dim)

    # Transform all blocks into body frame
    blocks_body = express_in_body_frame(blocks, p0s, se2_idxs)

    # Compute mean/var in batch
    mean_block = jnp.mean(blocks_body, axis=(0, 1))
    var_block = jnp.var(blocks_body, axis=(0, 1))

    # Global mean/var across all raw poses
    all_poses = jnp.concatenate(episode_list, axis=0)
    mean = jnp.mean(all_poses, axis=0)
    var = jnp.var(all_poses, axis=0)

    # Overwrite SE(2) parts with body-frame stats
    mean = mean.at[idxs].set(mean_block[idxs])
    var = var.at[idxs].set(var_block[idxs])

    return {
        "mean": mean,
        "std": jnp.sqrt(var),
    }


@jax.jit
def merge_mean_var(mean_A, var_A, n_A, mean_B, var_B, n_B):
    delta = mean_B - mean_A
    n_total = n_A + n_B

    mean = (n_A * mean_A + n_B * mean_B) / n_total
    M2 = n_A * var_A + n_B * var_B + (delta**2) * n_A * n_B / n_total
    var = M2 / n_total

    return mean, var, n_total


@jax.jit
def wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to the range [-pi, pi]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

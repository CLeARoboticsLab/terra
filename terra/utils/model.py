from terra.modeling import CONFIG_FILE_NAME
from terra.modeling import models
from terra.modeling import types
import jax.numpy as jnp
import jax
import numpy as np
import flax.nnx as nnx
from omegaconf import OmegaConf, DictConfig
import os
import orbax.checkpoint as ocp
from typing import Tuple, Any


def load_model_from_ckpt(
    ckpt_dir: str,
    step: int | None = None,  # if none, load the latest
) -> Tuple[nnx.Module, DictConfig, int]:
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist. ")

    model_cfg = OmegaConf.load(os.path.join(ckpt_dir, CONFIG_FILE_NAME))
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    dummy_scaler_params = make_dummy_scaler_params(model_cfg)
    model = models.get_model(model_cfg, dummy_scaler_params, nnx.Rngs(0))
    graphdef, model_state = nnx.split(model)

    with ocp.CheckpointManager(
        ckpt_dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as mngr:
        if step is None:
            step = mngr.latest_step()
        restored_pure_dict = mngr.restore(
            step,
            args=ocp.args.StandardRestore(item=model_state.to_pure_dict()),
        )

    model_state.replace_by_pure_dict(restored_pure_dict)
    model = nnx.merge(graphdef, model_state)

    return model, model_cfg, step


def batch_to_model_in_out(
    batch: Tuple[dict[str, types.ModelInputs], dict[str, types.ModelOutputs]],
) -> Tuple[types.ModelInputs, types.ModelOutputs]:
    model_in = _numpy_dict_to_jax(batch["model_in"])
    model_out = _numpy_dict_to_jax(batch["model_out"])
    return model_in, model_out


def repeat_along_batch_dim(x, B):
    return jax.tree.map(lambda y: jnp.tile(y[None, ...], (B,) + (1,) * y.ndim), x)


def make_dummy_scaler_params(cfg: dict):
    struct = cfg["scaler_params_struct"]
    dummy_scalar_params = {}
    for k, dim in struct.items():
        dummy_scalar_params[k] = {
            "mean": jnp.zeros(dim, dtype=jnp.float32),
            "std": jnp.ones(dim, dtype=jnp.float32),
        }
    return dummy_scalar_params


def array_from_types_configs(
    vector: np.ndarray, types_configs: list[dict]
) -> np.ndarray:
    """
    Extracts an array from the given array using the types configs.

    Args:
        array (np.ndarray): A numpy array representing the full array.
        types_configs (list[dict]): A list of dictionaries, each containing
                                     'idx' and 'dim' fields to extract slices from `array`.

    Returns:
        np.ndarray: A numpy array of the concatenated features.
    """
    slices = []
    for config in types_configs:
        idx = config["idx"]
        dim = config["dim"]
        slices.append(vector[..., idx : idx + dim])
    return np.concatenate(slices, axis=-1)


def obs_from_state(state: np.ndarray, sys_cfg: dict):
    return array_from_types_configs(state, sys_cfg["obs"]["types"])


def pose_from_system_pose(system_pose: np.ndarray, sys_cfg: dict):
    return array_from_types_configs(system_pose, sys_cfg["poses"]["types"])


def count_params(model: nnx.Module, filter=None):
    full_state = nnx.state(model)
    if filter is None:
        sel_state = full_state
    else:
        sel_state = full_state.filter(filter)
    sizes = jax.tree_util.tree_leaves(jax.tree.map(lambda x: x.size, sel_state))
    return int(sum(sizes))


def _numpy_dict_to_jax(numpy_dict: dict[Any, np.ndarray]) -> dict[Any, jnp.ndarray]:
    """Convert a dict of numpy arrays to a dict of JAX arrays."""
    return {k: jnp.array(v) for k, v in numpy_dict.items()}

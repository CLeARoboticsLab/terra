from terra.data import Names
from terra.modeling.types import ScalerParam
import json
import os
import jax.numpy as jnp
import jax


def save_scaler_params(scaler_params: dict[str, ScalerParam], save_dir: str):
    """
    Save scaler parameters to a JSON file in the specified dataset directory.

    Args:
        scaler_params (types.ScalerParams): Scaler parameters to save.
        save_dir (str): Directory where the scaler parameters will be saved.
    """
    data = jax.tree.map(lambda x: x.tolist(), scaler_params)
    os.makedirs(save_dir, exist_ok=True)
    scaler_params_path = os.path.join(save_dir, Names.SCALER_PARAMS_FILE_NAME)
    with open(scaler_params_path, "w") as f:
        json.dump(data, f, indent=4)


def load_scaler_params(load_dir: str) -> dict[str, ScalerParam]:
    """
    Load scaler parameters from a JSON file in the specified dataset directory.

    Args:
        load_dir (str): Directory where the scaler parameters are saved.

    Returns:
        dict[str, ScalerParam]: Scaler parameters loaded from the file.
    """
    scaler_params_path = os.path.join(load_dir, Names.SCALER_PARAMS_FILE_NAME)
    with open(scaler_params_path, "r") as f:
        data = json.load(f)
    return jax.tree.map(
        lambda x: jnp.array(x), data, is_leaf=lambda x: isinstance(x, list)
    )

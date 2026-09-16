def obs_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the observation dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The observation dimension.
    """
    ob_dims = [ob["dim"] for ob in sys_cfg["obs"]["types"]]
    return sum(ob_dims)


def pose_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the pose dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The pose dimension.
    """
    pose_dims = [pose["dim"] for pose in sys_cfg["poses"]["types"]]
    return sum(pose_dims)


def acts_dim_from_sys_config(sys_cfg: dict) -> int:
    """
    Extracts the action dimension from the system configuration.

    Args:
        cfg (dict): Configuration dictionary containing system parameters.

    Returns:
        int: The action dimension.
    """
    return sys_cfg["acts"]["dim"]

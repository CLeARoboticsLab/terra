import numbers

from terra.modeling.types import ModelInputs, ModelOutputs
from terra.utils.math import wrap_angle
import jax.numpy as jnp
import numpy as np


class RewardFunctionBase:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @staticmethod
    def model_io_to_reward_fn_inputs(*args, **kwargs) -> tuple:
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(
        self, *args, **kwargs
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        raise NotImplementedError("Subclasses must implement this method.")


class ErnestRewardFunction(RewardFunctionBase):
    # pose indices
    X_IDX = 0
    Y_IDX = 1
    BEARING_IDX = 2
    YAW_IDX = 3
    ROLL_IDX = 4

    # obs indices
    OBS_ROLL_IDX = 0
    WHEEL_FORCE_START_IDX = 2
    WHEEL_FORCE_DIM = 4 * 6

    # roll-limit huber penalty (frozen, not user-configurable)
    ROLL_LIMIT_HUBER_DELTA = 0.25
    ROLL_LIMIT_NONFINITE_EXCESS = 1.0
    DEFAULT_WEIGHT_ROLL_LIMIT = 0.0
    DEFAULT_ROLL_LIMIT_DEG = 15.0

    # action indices
    YAW_ACT_IDX = 1
    ROLL_ACT_IDX = 2

    # path line definition indices
    DESIRED_BEARING_IDX = 0
    A_IDX = 1
    B_IDX = 2
    C_IDX = 3

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.weight_delta_x = cfg.get("weight_delta_x", 10.0)
        self.weight_y = cfg.get("weight_y", 2.5)
        self.weight_bearing_dev = cfg.get("weight_bearing_dev", 0.25)
        self.weight_yaw = cfg.get("weight_yaw", 0.5)
        self.weight_roll = cfg.get("weight_roll", 1.0)
        self.weight_yaw_action = cfg.get("weight_yaw_action", 0.5)
        self.weight_roll_action = cfg.get("weight_roll_action", 2.0)
        self.weight_wheel_forces = cfg.get("weight_wheel_forces", 0.001)
        self.wheel_forces_huber_width = cfg.get("wheel_forces_huber_width", 1.0)
        self.huber_for_lateral_dev = cfg.get("huber_for_lateral_dev", False)
        self.delta_t = cfg.get("delta_t", 0.5)

        self.weight_roll_limit = self._validate_scalar(
            cfg.get("weight_roll_limit", self.DEFAULT_WEIGHT_ROLL_LIMIT),
            "weight_roll_limit",
            strictly_positive=False,
        )
        self.roll_limit_deg = self._validate_scalar(
            cfg.get("roll_limit_deg", self.DEFAULT_ROLL_LIMIT_DEG),
            "roll_limit_deg",
            strictly_positive=True,
        )
        self.roll_limit_rad = float(np.deg2rad(self.roll_limit_deg))

    @staticmethod
    def _validate_scalar(value, name: str, strictly_positive: bool) -> float:
        """Validate a real, finite reward scalar, rejecting booleans and nonnumerics."""
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a real scalar, got {value!r}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if strictly_positive:
            if value <= 0.0:
                raise ValueError(f"{name} must be strictly positive, got {value!r}")
        elif value < 0.0:
            raise ValueError(f"{name} must be nonnegative, got {value!r}")
        return value

    @staticmethod
    def model_io_to_reward_fn_inputs(
        path_def: jnp.ndarray,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
    ):
        next_poses = model_outputs["poses"]
        next_obs = model_outputs["obs"]
        actions = model_inputs["fut_acts"]
        poses = jnp.concatenate(
            [model_inputs["pose_hist"][:, -1:], next_poses[:, :-1]], axis=1
        )
        return path_def, poses, actions, next_poses, next_obs

    def __call__(
        self,
        path_def: jnp.ndarray,
        pose: jnp.ndarray,
        action: jnp.ndarray,
        next_pose: jnp.ndarray,
        next_obs: jnp.ndarray,
        **kwargs,
    ):
        bearing_dev = (
            next_pose[..., self.BEARING_IDX] - path_def[..., self.DESIRED_BEARING_IDX]
        )
        bearing_dev = wrap_angle(bearing_dev)

        fwd_advance_rew = (
            self._get_forward_advance(path_def, pose, next_pose) / self.delta_t
        )
        is_pos_adv = fwd_advance_rew > 0.0
        is_not_facing_forward = jnp.abs(bearing_dev) > (jnp.pi / 2)
        fwd_advance_rew = jnp.where(
            jnp.logical_and(is_pos_adv, is_not_facing_forward),
            -fwd_advance_rew,
            fwd_advance_rew,
        )

        lat_dev = self._get_lateral_deviation(path_def, next_pose)
        if self.huber_for_lateral_dev:
            lat_dev_pen = -self._huber(lat_dev, 1.0)
        else:
            lat_dev_pen = -(lat_dev**2)

        bearing_dev_pen = -(bearing_dev**2)

        yaw_pen = -(next_pose[..., self.YAW_IDX] ** 2)
        roll_pen = -(next_pose[..., self.ROLL_IDX] ** 2)

        yaw_action_pen = -(action[..., self.YAW_ACT_IDX] ** 2)
        roll_action_pen = -(action[..., self.ROLL_ACT_IDX] ** 2)

        wheel_forces = next_obs[
            ...,
            self.WHEEL_FORCE_START_IDX : self.WHEEL_FORCE_START_IDX
            + self.WHEEL_FORCE_DIM,
        ]
        wheel_forces_x = wheel_forces[..., 0::6]
        wheel_forces_y = wheel_forces[..., 1::6]
        wheel_force_pen = -jnp.sum(
            self._huber(wheel_forces_x, self.wheel_forces_huber_width)
            + self._huber(wheel_forces_y, self.wheel_forces_huber_width),
            axis=-1,
        )

        roll_limit_pen = self._roll_limit_penalty(next_obs[..., self.OBS_ROLL_IDX])
        if self.weight_roll_limit == 0.0:
            # Exact zeros so pathological roll values cannot leak in via 0 * inf.
            weighted_roll_limit_pen = jnp.zeros_like(roll_limit_pen)
        else:
            weighted_roll_limit_pen = self.weight_roll_limit * roll_limit_pen

        reward = (
            self.weight_delta_x * fwd_advance_rew
            + self.weight_y * lat_dev_pen
            + self.weight_bearing_dev * bearing_dev_pen
            + self.weight_yaw * yaw_pen
            + self.weight_roll * roll_pen
            + self.weight_yaw_action * yaw_action_pen
            + self.weight_roll_action * roll_action_pen
            + self.weight_wheel_forces * wheel_force_pen
            + weighted_roll_limit_pen
        )

        reward_comps = {
            "forward_advance": self.weight_delta_x * fwd_advance_rew,
            "lateral_deviation": self.weight_y * lat_dev_pen,
            "bearing_deviation": self.weight_bearing_dev * bearing_dev_pen,
            "yaw": self.weight_yaw * yaw_pen,
            "roll": self.weight_roll * roll_pen,
            "yaw_action": self.weight_yaw_action * yaw_action_pen,
            "roll_action": self.weight_roll_action * roll_action_pen,
            "wheel_forces": self.weight_wheel_forces * wheel_force_pen,
            "roll_limit": weighted_roll_limit_pen,
        }

        reward_comps_unweighted = {
            "forward_advance": fwd_advance_rew,
            "lateral_deviation": lat_dev_pen,
            "bearing_deviation": bearing_dev_pen,
            "yaw": yaw_pen,
            "roll": roll_pen,
            "yaw_action": yaw_action_pen,
            "roll_action": roll_action_pen,
            "wheel_forces": wheel_force_pen,
            "roll_limit": roll_limit_pen,
        }

        return reward, reward_comps, reward_comps_unweighted

    def _get_forward_advance(
        self, path_def: jnp.ndarray, pose: jnp.ndarray, next_pose: jnp.ndarray
    ):
        delta_x = next_pose[..., self.X_IDX] - pose[..., self.X_IDX]
        delta_y = next_pose[..., self.Y_IDX] - pose[..., self.Y_IDX]
        return path_def[..., self.B_IDX] * delta_x - path_def[..., self.A_IDX] * delta_y

    def _get_lateral_deviation(self, path_def: jnp.ndarray, next_pose: jnp.ndarray):
        return (
            path_def[..., self.A_IDX] * next_pose[..., self.X_IDX]
            + path_def[..., self.B_IDX] * next_pose[..., self.Y_IDX]
            + path_def[..., self.C_IDX]
        )

    def _roll_limit_penalty(self, roll: jnp.ndarray) -> jnp.ndarray:
        """Unweighted soft roll-limit penalty: -Huber(normalized exceedance)."""
        is_finite = jnp.isfinite(roll)
        # Sanitize before arithmetic so NaN/inf cannot leak into gradients.
        safe_roll = jnp.where(is_finite, roll, 0.0)
        excess = jnp.maximum(jnp.abs(safe_roll) / self.roll_limit_rad - 1.0, 0.0)
        excess = jnp.where(
            is_finite,
            excess,
            jnp.asarray(self.ROLL_LIMIT_NONFINITE_EXCESS, dtype=excess.dtype),
        )
        return -self._huber(excess, self.ROLL_LIMIT_HUBER_DELTA)

    def _huber(self, x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """Huber function for x with width w"""
        return jnp.where(jnp.abs(x) < w, 0.5 * x**2, w * (jnp.abs(x) - 0.5 * w))

"""Optional Gaussian sensor-noise injection for MPPI controllers.

The feature models a noisy incoming sensor stream: every distilled observation
and (absolute) pose measurement is perturbed once, with independent zero-mean
Gaussian draws whose per-feature standard deviations are

    noise_std[i] = level * dynamics_dataset_std[i]

where ``dynamics_dataset_std`` comes from the pretrained world model's
scaler parameters. The feature is entirely opt-in: when the config block is
absent or disabled, controllers never construct an injector, never read scaler
parameters, and never allocate a random generator.
"""

import numpy as np
import numbers
from collections.abc import Mapping
from typing import Optional


DEFAULT_LEVEL = 0.05
DEFAULT_SEED = 0

_OBS_KEY = "obs"
_POSE_KEY = "poses"


def sensor_noise_enabled(sensor_noise_cfg) -> bool:
    """Returns whether the (possibly absent) sensor-noise config is enabled.

    A missing config, ``None``, or an empty block is equivalent to disabled.
    Only the ``enabled`` field is inspected so that the disabled path never
    touches the rest of the configuration.
    """
    if sensor_noise_cfg is None:
        return False
    if not isinstance(sensor_noise_cfg, Mapping):
        raise TypeError(
            "sensor_noise_cfg must be a mapping or None, got "
            f"{type(sensor_noise_cfg).__name__}."
        )
    if "enabled" not in sensor_noise_cfg:
        return False
    enabled = sensor_noise_cfg["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(
            f"sensor_noise.enabled must be a bool, got {type(enabled).__name__}."
        )
    return enabled


def scaler_stds_from_model(model) -> dict[str, np.ndarray]:
    """Snapshots the observation/pose dataset standard deviations of a model.

    The values are copied into host NumPy arrays so that no mutable alias to
    the model variables is retained. Call this on the *base* world model,
    before any exploration or online-adaptation wrapper is applied.
    """
    scaler = getattr(model, "scaler", None)
    if scaler is None or getattr(scaler, "scaler_params", None) is None:
        raise ValueError(
            "sensor noise is enabled but the world model has no scaler "
            "parameters to derive per-feature noise magnitudes from."
        )
    scaler_params = scaler.scaler_params
    stds = {}
    for key in (_OBS_KEY, _POSE_KEY):
        if key not in scaler_params:
            raise ValueError(
                f"sensor noise is enabled but the world model scaler has no "
                f"'{key}' entry; found {sorted(scaler_params.keys())}."
            )
        entry = scaler_params[key]
        if not isinstance(entry, Mapping) or "std" not in entry:
            raise ValueError(
                f"sensor noise is enabled but the world model scaler entry "
                f"'{key}' has no 'std' value."
            )
        std = entry["std"]
        std = getattr(std, "value", std)  # unwrap nnx.Variable
        stds[key] = np.array(std, dtype=np.float64, copy=True)
    return stds


def make_sensor_noise_injector(
    sensor_noise_cfg,
    scaler_stds: dict[str, np.ndarray],
    obs_dim: int,
    pose_dim: int,
    yaw_idx: int,
) -> Optional["SensorNoiseInjector"]:
    """Builds an injector, or returns None when sensor noise is disabled."""
    if not sensor_noise_enabled(sensor_noise_cfg):
        return None

    level = _validate_level(sensor_noise_cfg.get("level", DEFAULT_LEVEL))
    seed = _validate_seed(sensor_noise_cfg.get("seed", DEFAULT_SEED))
    obs_std = _validate_std(scaler_stds.get(_OBS_KEY), _OBS_KEY, obs_dim)
    pose_std = _validate_std(scaler_stds.get(_POSE_KEY), _POSE_KEY, pose_dim)

    if not isinstance(yaw_idx, numbers.Integral) or isinstance(yaw_idx, bool):
        raise TypeError(f"se2 yaw index must be an int, got {yaw_idx!r}.")
    yaw_idx = int(yaw_idx)
    if not 0 <= yaw_idx < pose_dim:
        raise ValueError(
            f"se2 yaw index {yaw_idx} is out of range for pose dimension {pose_dim}."
        )

    return SensorNoiseInjector(
        level=level,
        seed=seed,
        obs_std=obs_std,
        pose_std=pose_std,
        yaw_idx=yaw_idx,
    )


class SensorNoiseInjector:
    """Perturbs distilled observation/pose measurements with Gaussian noise.

    Owns a dedicated ``numpy.random.Generator`` so that the corruption process
    is reproducible from ``sensor_noise.seed`` and never consumes or perturbs
    the planner/adaptation JAX RNG streams. The generator is seeded once at
    construction and is deliberately *not* reseeded on controller reset, so
    repeated episodes within one process see distinct draws while the whole run
    stays reproducible.
    """

    def __init__(
        self,
        level: float,
        seed: int,
        obs_std: np.ndarray,
        pose_std: np.ndarray,
        yaw_idx: int,
    ):
        self.level = level
        self.seed = seed
        self.obs_std = obs_std
        self.pose_std = pose_std
        self.yaw_idx = yaw_idx
        self.obs_noise_std = level * obs_std
        self.pose_noise_std = level * pose_std
        self.rng = np.random.default_rng(seed)

    def perturb(self, item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Returns a copy of `item` with noisy `obs` and `pose` entries.

        Exactly one observation draw and one pose draw are consumed per call;
        all other entries (actions, path line definition, ...) are passed
        through untouched, and the input arrays are never mutated.
        """
        noisy_item = dict(item)
        noisy_item["obs"] = self._perturb_obs(item["obs"])
        noisy_item["pose"] = self._perturb_pose(item["pose"])
        return noisy_item

    def _perturb_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        self._check_shape(obs, self.obs_noise_std, "obs")
        noise = self.rng.standard_normal(self.obs_noise_std.shape)
        return (obs + noise * self.obs_noise_std).astype(obs.dtype, copy=False)

    def _perturb_pose(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose)
        self._check_shape(pose, self.pose_noise_std, "pose")
        noise = self.rng.standard_normal(self.pose_noise_std.shape)
        noisy = pose + noise * self.pose_noise_std
        # Only the periodic global yaw is wrapped; nothing else is clipped, so
        # every other feature keeps the requested Gaussian distribution.
        noisy[self.yaw_idx] = wrap_angle(noisy[self.yaw_idx])
        return noisy.astype(pose.dtype, copy=False)

    @staticmethod
    def _check_shape(value: np.ndarray, std: np.ndarray, name: str):
        if value.shape != std.shape:
            raise ValueError(
                f"sensor noise expected {name} of shape {std.shape}, got "
                f"{value.shape}."
            )


def wrap_angle(angle):
    """Wrap angles to the range [-pi, pi] (NumPy counterpart of utils.math)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _validate_level(level) -> float:
    if isinstance(level, bool) or not isinstance(level, numbers.Real):
        raise TypeError(
            f"sensor_noise.level must be a real scalar, got {level!r}."
        )
    level = float(level)
    if not np.isfinite(level):
        raise ValueError(f"sensor_noise.level must be finite, got {level}.")
    if level < 0.0:
        raise ValueError(f"sensor_noise.level must be nonnegative, got {level}.")
    return level


def _validate_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"sensor_noise.seed must be an integer, got {seed!r}.")
    return int(seed)


def _validate_std(std, name: str, expected_dim: int) -> np.ndarray:
    if std is None:
        raise ValueError(
            f"sensor noise is enabled but no '{name}' scaler standard "
            "deviations were provided."
        )
    std = np.asarray(std, dtype=np.float64)
    if std.ndim != 1:
        raise ValueError(
            f"'{name}' scaler std must be a 1-D vector, got shape {std.shape}."
        )
    if std.shape[0] != expected_dim:
        raise ValueError(
            f"'{name}' scaler std has dimension {std.shape[0]} but the "
            f"distilled measurement has dimension {expected_dim}."
        )
    if not np.all(np.isfinite(std)):
        bad = np.flatnonzero(~np.isfinite(std)).tolist()
        raise ValueError(
            f"'{name}' scaler std has non-finite entries at indices {bad}."
        )
    if np.any(std < 0.0):
        bad = np.flatnonzero(std < 0.0).tolist()
        raise ValueError(
            f"'{name}' scaler std has negative entries at indices {bad}."
        )
    return std

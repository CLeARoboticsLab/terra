from typing import TypedDict
from jax import numpy as jnp


class ModelInputs(TypedDict):
    pass


class ModelOutputs(TypedDict):
    pass


class ScalerParam(TypedDict):
    mean: jnp.ndarray
    std: jnp.ndarray


class ScalerParams(TypedDict):
    pass


ScalerParamsMapping = dict[str, str]


class WorldModel:
    class TrainingInputs(ModelInputs):
        obs_hist: jnp.ndarray  # observation history from t-H to t (B, H+1, obs_dim)
        pose_hist: jnp.ndarray  # pose history from t-H to t (B, H+1, pose_dim)
        acts_hist: jnp.ndarray  # action history from t-H to t-1 (B, H, act_dim)
        fut_acts: jnp.ndarray  # future actions to apply from t to t+T-1 (B, T, act_dim)

    class TrainingOutputs(ModelOutputs):
        obs: jnp.ndarray  # future observations, t+1 to t+T (B, T, obs_dim)
        poses: jnp.ndarray  # future poses (body), t+1 to t+T (B, T, pose_dim)
        pose_hist: jnp.ndarray  # same as above, just for putting pred in body frame

    class InferenceInputs(ModelInputs):
        obs_hist: jnp.ndarray  # observation history from t-H to t (B, H+1, obs_dim)
        pose_hist: jnp.ndarray  # pose history from t-H to t (B, H+1, pose_dim)
        acts_hist: jnp.ndarray  # action history from t-H to t-1 (B, H, act_dim)
        fut_acts: jnp.ndarray  # future actions to apply from t to t+T-1 (B, T, act_dim)

    class InferenceOutputs(ModelOutputs):
        obs: jnp.ndarray  # future observations, t+1 to t+T (B, T, obs_dim)
        poses: jnp.ndarray  # future poses (world), t+1 to t+T (B, T, pose_dim)

    class ScalerParams(ScalerParams):
        obs: ScalerParam
        acts: ScalerParam
        poses: ScalerParam

    scaler_params_mapping: ScalerParamsMapping = {
        "obs_hist": "obs",
        "pose_hist": "poses",
        "acts_hist": "acts",
        "fut_acts": "acts",
        "obs": "obs",
        "poses": "poses",
    }


class BehaviorCloningWithPoseHist:
    class TrainingInputs(ModelInputs):
        obs_hist: jnp.ndarray  # observation history from t-H to t (B, H+1, obs_dim)
        pose_hist: jnp.ndarray  # pose history from t-H to t (B, H+1, pose_dim)
        acts_hist: jnp.ndarray  # action history from t-H to t-1 (B, H, act_dim)

    class TrainingOutputs(ModelOutputs):
        actions: jnp.ndarray  # future actual actions, t to t+T-1 (B, T, act_dim)

    class InferenceInputs(ModelInputs):
        obs_hist: jnp.ndarray  # observation history from t-H to t (B, H+1, obs_dim)
        pose_hist: jnp.ndarray  # pose history from t-H to t (B, H+1, pose_dim)
        acts_hist: jnp.ndarray  # action history from t-H to t-1 (B, H, act_dim)

    class InferenceOutputs(ModelOutputs):
        actions: jnp.ndarray  # future actions, t to t+T-1 (B, T, act_dim)

    class ScalerParams(ScalerParams):
        obs: ScalerParam
        acts: ScalerParam
        poses: ScalerParam

    scaler_params_mapping: ScalerParamsMapping = {
        "obs_hist": "obs",
        "pose_hist": "poses",
        "acts_hist": "acts",
        "actions": "acts",
    }

from terra.modeling import types, models
import flax.nnx as nnx
import jax.numpy as jnp
from typing import Tuple
import jax
import optax


class Loss:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.loss_cfg: dict = cfg["loss"]
        self.weights: dict = self.loss_cfg["weights"]

    @property
    def loss_terms(self) -> list[str]:
        return list(self.weights.keys())

    def __call__(
        self,
        model: nnx.Module,
        x: types.ModelInputs,
        y: types.ModelOutputs,
        deterministic: bool,
        aux_losses: dict = {},
        **kwargs,
    ) -> Tuple[jnp.ndarray, dict]:
        losses, _ = self.compute_losses(model, x, y, deterministic, **kwargs)
        losses.update(aux_losses)
        assert set(losses.keys()) == set(self.loss_terms), (
            "Losses do not match weights. "
            f"Got {losses.keys()} losses and {self.weights.keys()} weights. "
            "Check the loss config file and add or remove weight terms as needed."
        )
        weighted_losses = {
            k: self.weights[k] * jnp.mean(losses[k]) for k in self.loss_terms
        }
        loss = sum(weighted_losses.values())
        return loss, weighted_losses

    def compute_losses(
        self,
        model: nnx.Module,
        x: types.ModelInputs,
        y: types.ModelOutputs,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[dict, dict]:
        """Returns a dict of losses (shape (B,)) with keys matching the weights in
        the config file, and a dict of extra info"""
        raise NotImplementedError("Must implement compute_losses method")


class WorldModelLoss(Loss):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.pose_weights = jnp.array(1.0)
        self.obs_weights = jnp.array(1.0)
        if "component_weights" in self.loss_cfg:
            if "poses" in self.loss_cfg["component_weights"]:
                self.pose_weights = jnp.array(
                    self.loss_cfg["component_weights"]["poses"]
                )
            if "obs" in self.loss_cfg["component_weights"]:
                self.obs_weights = jnp.array(self.loss_cfg["component_weights"]["obs"])

    def compute_losses(
        self,
        model: models.WorldModel,
        x: types.WorldModel.TrainingInputs,
        y: types.WorldModel.TrainingOutputs,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[dict, dict]:
        y_pred = model(x, deterministic=deterministic)
        # put actual poses in body frame and scale
        y_scaled = jax.lax.stop_gradient(model._process_inputs(y))
        obs_error = y_pred["obs"] - y_scaled["obs"]
        poses_error = y_pred["poses"] - y_scaled["poses"]
        obs_loss = (obs_error**2) * self.obs_weights
        poses_loss = (poses_error**2) * self.pose_weights
        obs_loss_mean = _mean_except_batch(obs_loss)
        poses_loss_mean = _mean_except_batch(poses_loss)
        loss_dict = {
            "obs_loss": obs_loss_mean,
            "poses_loss": poses_loss_mean,
        }
        return loss_dict, {
            "obs_error": obs_error,
            "poses_error": poses_error,
            "obs_loss": obs_loss,
            "poses_loss": poses_loss,
            "y_pred": y_pred,
        }


class WorldModelAdaptationLoss(WorldModelLoss):
    def compute_losses(
        self,
        model: models.WorldModel,
        x: types.WorldModel.TrainingInputs,
        y: types.WorldModel.TrainingOutputs,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[dict, dict]:
        loss_dict, extras = super().compute_losses(model, x, y, deterministic, **kwargs)

        if "grad_reward_wrt_pose" in kwargs and "grad_reward_wrt_obs" in kwargs:
            grad_reward_wrt_pose = kwargs["grad_reward_wrt_pose"]
            poses_error = extras["poses_error"]
            reward_scaled_loss = jnp.sum(
                (grad_reward_wrt_pose * poses_error) ** 2, axis=-1
            )

            grad_reward_wrt_obs = kwargs["grad_reward_wrt_obs"]
            obs_error = extras["obs_error"]
            reward_scaled_loss += jnp.sum(
                (grad_reward_wrt_obs * obs_error) ** 2, axis=-1
            )

            loss_dict["reward_scaled_loss"] = _mean_except_batch(reward_scaled_loss)
            extras["reward_scaled_loss"] = reward_scaled_loss

        if "norm_deltaW" in kwargs:
            loss_dict["norm_deltaW_loss"] = kwargs["norm_deltaW"]

        return loss_dict, extras


def loss_pred_classification_loss_per_step(
    loss_pred_logits: jnp.ndarray,
    dyn_loss: jnp.ndarray,
    thresholds: jnp.ndarray,
    class_weights: jnp.ndarray | None,
) -> jnp.ndarray:
    """Per-(batch, timestep) cross entropy against the binned dynamics loss."""
    # digitize returns indices of the bins to which each value belongs.
    # Shape: (B, T) with integer values [0, NumThresholds]
    target_bins = jnp.digitize(dyn_loss, thresholds)
    loss_per_step = optax.softmax_cross_entropy_with_integer_labels(
        logits=loss_pred_logits, labels=target_bins
    )
    if class_weights is not None:
        weights = class_weights[target_bins]
        weights = weights / (jnp.mean(weights) + 1e-8)
        loss_per_step = loss_per_step * weights
    return loss_per_step


class WorldModelAdaptationLossPredLoss(WorldModelAdaptationLoss):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.thresholds = jnp.array(self.loss_cfg["loss_pred_thresholds"])
        self.loss_pred_class_weights = None
        if "loss_pred_class_weights" in self.loss_cfg:
            loss_pred_class_weights = self.loss_cfg["loss_pred_class_weights"]
            if not _is_none_str(loss_pred_class_weights):
                self.loss_pred_class_weights = jnp.array(loss_pred_class_weights)
                expected_num_bins = self.thresholds.shape[0] + 1
                print(
                    f"Using loss_pred_class_weights: {self.loss_pred_class_weights}",
                    flush=True,
                )
                if self.loss_pred_class_weights.shape != (expected_num_bins,):
                    raise ValueError(
                        "loss_pred_class_weights must have one weight per loss_pred "
                        f"bin. Expected {expected_num_bins}, got "
                        f"{self.loss_pred_class_weights.shape[0]}."
                    )

    def compute_losses(
        self,
        model: models.WorldModelWithLossPredBase,
        x: types.WorldModel.TrainingInputs,
        y: types.WorldModel.TrainingOutputs,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[dict, dict]:
        loss_dict, extras = super().compute_losses(model, x, y, deterministic, **kwargs)

        y_pred = extras["y_pred"]
        if "loss_pred_logits" not in y_pred:
            raise ValueError(
                "Model output 'y' does not contain 'loss_pred_logits'. "
                "Ensure you are wrapping the model with WorldModelWithLossPredBase."
            )
        loss_pred_logits = y_pred["loss_pred_logits"]

        obs_loss = jnp.mean(extras["obs_loss"], axis=-1)  # Shape: (B, T)
        poses_loss = jnp.mean(extras["poses_loss"], axis=-1)  # Shape: (B, T)
        dyn_loss = (
            self.weights["obs_loss"] * obs_loss
            + self.weights["poses_loss"] * poses_loss
        )
        if "reward_scaled_loss" in extras:
            reward_scaled_loss = extras["reward_scaled_loss"]
            dyn_loss += self.weights["reward_scaled_loss"] * reward_scaled_loss

        dyn_loss = jax.lax.stop_gradient(dyn_loss)
        loss_pred_loss_per_step = loss_pred_classification_loss_per_step(
            loss_pred_logits,
            dyn_loss,
            self.thresholds,
            self.loss_pred_class_weights,
        )

        loss_pred_loss = _mean_except_batch(loss_pred_loss_per_step)
        loss_dict["loss_pred_loss"] = loss_pred_loss

        return loss_dict, extras


class BehaviorCloningWithPoseHistLoss(Loss):
    def compute_losses(
        self,
        model: models.BehaviorCloningWithPoseHist,
        x: types.BehaviorCloningWithPoseHist.TrainingInputs,
        y: types.BehaviorCloningWithPoseHist.TrainingOutputs,
        deterministic: bool,
        **kwargs,
    ) -> Tuple[dict, dict]:
        y_pred = model(x, deterministic=deterministic)
        y_scaled = jax.lax.stop_gradient(model.scaler.scale(y))
        act_error = y_pred["actions"] - y_scaled["actions"]
        act_loss = _mean_except_batch(act_error**2)
        loss_dict = {"actions_loss": act_loss}
        return loss_dict, {}


def _mean_except_batch(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the mean of an array over all dimensions except the first one."""
    if x.ndim == 1:
        return x
    else:
        return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def _is_none_str(value) -> bool:
    return isinstance(value, str) and value.lower() == "none"


def get_loss(cfg: dict) -> Loss:
    losses = {
        "world_model": WorldModelLoss,
        "world_model_adaptation": WorldModelAdaptationLoss,
        "world_model_adaptation_loss_pred": WorldModelAdaptationLossPredLoss,
        "behavior_cloning_with_pose_hist": BehaviorCloningWithPoseHistLoss,
    }
    loss_type = cfg["loss"]["type"]
    if loss_type not in losses:
        raise ValueError(f"Loss type '{loss_type}' is not supported.")
    return losses[loss_type](cfg)

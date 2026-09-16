import terra.modeling.losses as losses
import jax.numpy as jnp
import numpy as np
import pytest


B, T, OBS_DIM, POSE_DIM = 2, 3, 2, 2

OBS_WEIGHTS = [0.5, 2.0]
POSE_WEIGHTS = [3.0, 0.25]
WEIGHTS = {
    "obs_loss": 10.0,
    "poses_loss": 7.0,
    "reward_scaled_loss": 0.05,
    "loss_pred_loss": 20.0,
}


class _StubLossPredModel:
    """Minimal stand-in for a loss-predictor-wrapped world model.

    The loss only needs the model to return predictions and to map the ground
    truth into the same (already scaled, body-frame) space.
    """

    def __init__(self, y_pred: dict):
        self.y_pred = y_pred

    def __call__(self, x, deterministic=None):
        return self.y_pred

    def _process_inputs(self, y):
        return y


def _loss_pred_loss_cfg(**extra) -> dict:
    loss_cfg = {
        "type": "world_model_adaptation_loss_pred",
        "weights": dict(WEIGHTS),
        "component_weights": {"obs": OBS_WEIGHTS, "poses": POSE_WEIGHTS},
        "loss_pred_thresholds": [5.0, 10.0],
    }
    loss_cfg.update(extra)
    return {"loss": loss_cfg}


def _loss_pred_test_arrays():
    rng = np.random.default_rng(0)

    def arr(*shape):
        return jnp.array(rng.normal(size=shape), dtype=jnp.float32)

    x = {}
    y = {"obs": arr(B, T, OBS_DIM), "poses": arr(B, T, POSE_DIM)}
    y_pred = {
        "obs": arr(B, T, OBS_DIM),
        "poses": arr(B, T, POSE_DIM),
        "loss_pred_logits": jnp.array(rng.normal(size=(B, T, 3)), dtype=jnp.float32),
    }
    kwargs = {
        "grad_reward_wrt_pose": arr(B, T, POSE_DIM),
        "grad_reward_wrt_obs": arr(B, T, OBS_DIM),
    }
    return x, y, y_pred, kwargs


def _expected_world_model_loss_target(y, y_pred, kwargs) -> jnp.ndarray:
    """The unbinned, reward-weighted dynamics loss the loss predictor must predict."""
    obs_error = y_pred["obs"] - y["obs"]
    poses_error = y_pred["poses"] - y["poses"]
    obs_loss = (obs_error**2) * jnp.array(OBS_WEIGHTS)
    poses_loss = (poses_error**2) * jnp.array(POSE_WEIGHTS)
    reward_scaled_loss = jnp.sum(
        (kwargs["grad_reward_wrt_pose"] * poses_error) ** 2, axis=-1
    ) + jnp.sum((kwargs["grad_reward_wrt_obs"] * obs_error) ** 2, axis=-1)
    return (
        WEIGHTS["obs_loss"] * jnp.mean(obs_loss, axis=-1)
        + WEIGHTS["poses_loss"] * jnp.mean(poses_loss, axis=-1)
        + WEIGHTS["reward_scaled_loss"] * reward_scaled_loss
    )


def test_discrete_loss_pred_loss_matches_binned_target():
    x, y, y_pred, kwargs = _loss_pred_test_arrays()
    loss = losses.get_loss(_loss_pred_loss_cfg())
    assert isinstance(loss, losses.WorldModelAdaptationLossPredLoss)

    loss_dict, _ = _compute(loss, x, y, y_pred, kwargs)

    target = _expected_world_model_loss_target(y, y_pred, kwargs)
    target_bins = jnp.digitize(target, jnp.array([5.0, 10.0]))
    log_probs = y_pred["loss_pred_logits"] - jnp.log(
        jnp.sum(jnp.exp(y_pred["loss_pred_logits"]), axis=-1, keepdims=True)
    )
    expected = jnp.mean(
        -jnp.take_along_axis(log_probs, target_bins[..., None], axis=-1)[..., 0],
        axis=-1,
    )

    assert loss_dict["loss_pred_loss"].shape == (B,)
    assert jnp.allclose(loss_dict["loss_pred_loss"], expected, atol=1e-5)


def test_loss_pred_loss_requires_loss_pred_logits():
    x, y, y_pred, kwargs = _loss_pred_test_arrays()
    y_pred = dict(y_pred)
    del y_pred["loss_pred_logits"]
    loss = losses.get_loss(_loss_pred_loss_cfg())
    with pytest.raises(ValueError, match="loss_pred_logits"):
        _compute(loss, x, y, y_pred, kwargs)


def _compute(loss, x, y, y_pred, kwargs):
    return loss.compute_losses(
        _StubLossPredModel(y_pred), x, y, deterministic=True, **kwargs
    )

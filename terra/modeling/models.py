import terra.modeling.modules as modules
import terra.modeling.embedding as emb
import terra.modeling.types as types
import terra.modeling.scaler as scaler
import terra.utils.config as config_utils
from terra.utils.math import express_in_body_frame, express_in_world_frame
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Union


class ModelBase(nnx.Module):
    def __init__(
        self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs, **kwargs
    ):
        self.cfg = cfg
        self.model_cfg = cfg["model"]
        self.sys_cfg = cfg["system"]
        self.T = self.model_cfg["dataset"]["prediction_length"]
        self.embed_dim = self.model_cfg["embed_dim"]
        self.attention_dropout_rate = self.model_cfg["dropout"]["attention"]
        self.mlp_dropout_rate = self.model_cfg["dropout"]["mlp"]

    def __call__(
        self,
        x: types.ModelInputs,
        deterministic: bool | None = None,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement __call__ method")

    def inference(
        self,
        x: types.ModelInputs,
        **kwargs,
    ) -> types.ModelOutputs:
        raise NotImplementedError("Must implement inference method")


class WorldModel(ModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.se2_idxs: dict = cfg["system"]["poses"]["se2_idxs"]

        # input processing
        self.scaler = scaler.Scaler(
            scaler_params,
            types.WorldModel.scaler_params_mapping,
        )
        self.hist_embed = emb.HistoryEmbedding(cfg, rngs, embed_poses=True)
        h_size = self.embed_dim * cfg["model"]["embedding"]["mlp_hidden_size_factor"]
        self.fut_acts_embed = modules.Embedding(
            seq_len=self.T,
            input_dim=config_utils.acts_dim_from_sys_config(self.sys_cfg),
            hidden_dims=[h_size] * cfg["model"]["embedding"]["future_actions_layers"],
            embed_dim=self.embed_dim,
            rngs=rngs,
        )

        # dynamics head
        attn_cfg = self.model_cfg["dynamics"]["attention"]
        dec_cfg = self.model_cfg["dynamics"]["decoder"]
        dec_h_size = self.embed_dim * dec_cfg["mlp_hidden_size_factor"]
        self.dynamics = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.embed_dim,
            mlp_hidden_dim=self.embed_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.obs_dec = modules.MLP(
            input_dim=self.embed_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=config_utils.obs_dim_from_sys_config(self.sys_cfg),
            rngs=rngs,
        )
        self.poses_dec = modules.MLP(
            input_dim=self.embed_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=config_utils.pose_dim_from_sys_config(self.sys_cfg),
            rngs=rngs,
        )

    def __call__(
        self,
        x: types.WorldModel.TrainingInputs,
        deterministic: bool | None = None,
    ) -> types.WorldModel.TrainingOutputs:
        x = self._process_inputs(x)
        hist_emb = self.hist_embed(x, deterministic=deterministic)
        fut_acts_emb = self.fut_acts_embed(x["fut_acts"], deterministic=deterministic)
        latent_dyn_pred = self.dynamics(
            fut_acts_emb, hist_emb, deterministic=deterministic
        )
        obs = self.obs_dec(latent_dyn_pred, deterministic=deterministic)
        poses = self.poses_dec(latent_dyn_pred, deterministic=deterministic)
        return {
            "obs": obs,
            "poses": poses,  # body frame, scaled
            "hist_emb": hist_emb,
            "fut_acts_emb": fut_acts_emb,
        }

    def inference(
        self,
        x: types.WorldModel.InferenceInputs,
        **kwargs,
    ) -> types.WorldModel.InferenceOutputs:
        y = self(x, deterministic=True)
        y = self._process_outputs(x, y)
        return y

    def _process_inputs(
        self,
        x: Union[
            types.WorldModel.TrainingInputs,
            types.WorldModel.InferenceInputs,
        ],
    ):
        x = _body_frame_poses(x, self.se2_idxs)
        x = self.scaler.scale(x)
        return x

    def _process_outputs(
        self,
        x: Union[
            types.WorldModel.TrainingInputs,
            types.WorldModel.InferenceInputs,
        ],
        y: Union[
            types.WorldModel.TrainingOutputs,
            types.WorldModel.InferenceOutputs,
        ],
    ):
        y = self.scaler.unscale(y)
        y = _world_frame_poses(x, y, self.se2_idxs)
        return y


class WorldModelWithLossPredBase(ModelBase):
    """A world model wrapper that adds a loss predictor."""

    HIGHEST_LOSS_BIN_INIT = "highest_loss_bin"
    NONE_INIT = "none"

    def __init__(self, world_model: WorldModel, cfg: dict, rngs: nnx.Rngs):
        self.cfg = cfg
        self.world_model = world_model
        self.thresholds = self.cfg["loss_pred"]["thresholds"]
        self.num_bins = len(self.thresholds) + 1

    def __call__(
        self,
        x: types.WorldModel.TrainingInputs,
        deterministic: bool | None = None,
    ) -> types.WorldModel.TrainingOutputs:
        raise NotImplementedError("Must implement __call__ method")

    def inference(
        self,
        x: types.WorldModel.InferenceInputs,
        **kwargs,
    ) -> types.WorldModel.InferenceOutputs:
        y = self(x, deterministic=True)
        y = self._process_outputs(x, y)
        loss_pred_logits = y["loss_pred_logits"]
        loss_pred_probs = jax.nn.softmax(loss_pred_logits, axis=-1)
        bin_values = jnp.concatenate([jnp.array([0.0]), jnp.array(self.thresholds)])
        bin_values = bin_values.reshape(1, 1, -1)
        expected_model_loss = jnp.sum(loss_pred_probs * bin_values, axis=-1)
        y["exploration_reward"] = expected_model_loss
        return y

    @property
    def dynamics(self):
        return self.world_model.dynamics

    def _process_inputs(
        self,
        x: Union[
            types.WorldModel.TrainingInputs,
            types.WorldModel.InferenceInputs,
        ],
    ):
        return self.world_model._process_inputs(x)

    def _process_outputs(
        self,
        x: Union[
            types.WorldModel.TrainingInputs,
            types.WorldModel.InferenceInputs,
        ],
        y: Union[
            types.WorldModel.TrainingOutputs,
            types.WorldModel.InferenceOutputs,
        ],
    ):
        return self.world_model._process_outputs(x, y)

    def _maybe_init_loss_pred_output_layer(self, output_layer: nnx.Linear):
        init_cfg = self.cfg["loss_pred"].get("init")
        if init_cfg is None:
            return

        init_type = init_cfg.get("type")
        if init_type not in [self.NONE_INIT, self.HIGHEST_LOSS_BIN_INIT]:
            raise ValueError(f"Unknown loss_pred init type: {init_type}")

        if init_type == self.NONE_INIT:
            return

        prior_prob = init_cfg.get("prior_prob", 0.99)
        if not 0.0 < prior_prob <= 1.0:
            raise ValueError("loss_pred init prior_prob must be in (0, 1].")

        output_layer.kernel.value = jnp.zeros_like(output_layer.kernel.value)

        bias = jnp.zeros((self.num_bins,))
        if self.num_bins > 1:
            if prior_prob == 1.0:
                raise ValueError(
                    "loss_pred init prior_prob must be less than 1.0 when there are "
                    "multiple bins."
                )
            other_prob = (1.0 - prior_prob) / (self.num_bins - 1)
            logit_margin = jnp.log(prior_prob / other_prob)
            bias = bias.at[-1].set(logit_margin)
        output_layer.bias.value = bias
        print(
            f"Initialized loss_pred output layer with prior probability: {prior_prob}",
            flush=True,
        )


class WorldModelWithLossPredTransformer(WorldModelWithLossPredBase):
    """Loss predictor with a transformer architecture."""

    def __init__(self, world_model: WorldModel, cfg: dict, rngs: nnx.Rngs):
        super().__init__(world_model, cfg, rngs)

        attn_cfg = self.cfg["loss_pred"]["attention"]
        dec_cfg = self.cfg["loss_pred"]["decoder"]
        dec_h_size = self.world_model.embed_dim * dec_cfg["mlp_hidden_size_factor"]
        self.loss_pred = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.world_model.embed_dim,
            mlp_hidden_dim=self.world_model.embed_dim
            * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.world_model.attention_dropout_rate,
            mlp_dropout_rate=self.world_model.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.loss_pred_dec = modules.MLP(
            input_dim=self.world_model.embed_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=self.num_bins,
            rngs=rngs,
        )
        self._maybe_init_loss_pred_output_layer(self.loss_pred_dec.output_layer)

    def __call__(
        self,
        x: types.WorldModel.TrainingInputs,
        deterministic: bool | None = None,
    ) -> types.WorldModel.TrainingOutputs:
        y = self.world_model(x, deterministic=deterministic)
        hist_emb = jax.lax.stop_gradient(y["hist_emb"])
        fut_acts_emb = jax.lax.stop_gradient(y["fut_acts_emb"])
        loss_pred_out = self.loss_pred(
            fut_acts_emb, hist_emb, deterministic=deterministic
        )
        loss_pred_head = self.loss_pred_dec(loss_pred_out, deterministic=deterministic)
        return {
            **y,
            "loss_pred_logits": loss_pred_head,
        }


class BehaviorCloningWithPoseHist(ModelBase):
    def __init__(self, cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs):
        super().__init__(cfg, scaler_params, rngs)
        self.se2_idxs: dict = cfg["system"]["poses"]["se2_idxs"]

        self.scaler = scaler.Scaler(
            scaler_params,
            types.BehaviorCloningWithPoseHist.scaler_params_mapping,
        )
        self.hist_embed = emb.HistoryEmbedding(cfg, rngs, embed_poses=True)
        self.actions_queries = nnx.Param(
            jax.random.normal(rngs["params"](), (self.T, self.embed_dim)) * 0.02
        )

        attn_cfg = self.model_cfg["actions"]["attention"]
        dec_cfg = self.model_cfg["actions"]["decoder"]
        dec_h_size = self.embed_dim * dec_cfg["mlp_hidden_size_factor"]
        self.actions = modules.TransformerDecoder(
            num_layers=attn_cfg["layers"],
            embed_dim=self.embed_dim,
            mlp_hidden_dim=self.embed_dim * attn_cfg["mlp_hidden_size_factor"],
            num_heads=attn_cfg["heads"],
            rngs=rngs,
            attention_dropout_rate=self.attention_dropout_rate,
            mlp_dropout_rate=self.mlp_dropout_rate,
            mask=attn_cfg["mask"],
        )
        self.acts_dec = modules.MLP(
            input_dim=self.embed_dim,
            hidden_dims=[dec_h_size] * dec_cfg["layers"],
            output_dim=self.sys_cfg["acts"]["dim"],
            rngs=rngs,
        )

    def __call__(
        self,
        x: types.BehaviorCloningWithPoseHist.TrainingInputs,
        deterministic: bool | None = None,
    ) -> types.BehaviorCloningWithPoseHist.TrainingOutputs:
        B = x["obs_hist"].shape[0]
        x = self._process_inputs(x)
        hist_emb = self.hist_embed(x, deterministic=deterministic)
        actions_queries = jnp.broadcast_to(
            self.actions_queries[None, ...], (B, *self.actions_queries.shape)
        )
        acts = self.actions(actions_queries, hist_emb, deterministic=deterministic)
        acts = self.acts_dec(acts, deterministic=deterministic)
        return {
            "actions": acts,
        }

    def inference(
        self,
        x: types.BehaviorCloningWithPoseHist.InferenceInputs,
    ) -> types.BehaviorCloningWithPoseHist.InferenceOutputs:
        x = self(x, deterministic=True)
        x = self.scaler.unscale(x)
        return x

    def _process_inputs(
        self,
        x: Union[
            types.BehaviorCloningWithPoseHist.TrainingInputs,
            types.BehaviorCloningWithPoseHist.InferenceInputs,
        ],
    ):
        x = _body_frame_poses(x, self.se2_idxs)
        x = self.scaler.scale(x)
        return x


def _body_frame_poses(
    x: types.ModelInputs, se2_idxs: dict[str, int]
) -> types.ModelInputs:
    """Transforms the poses to the body frame"""
    pose_hist: jnp.ndarray = x["pose_hist"]
    p_t_H = pose_hist[:, 0]
    p_t = pose_hist[:, -1]

    transformed_x = {}
    for k, v in x.items():
        if k == "pose_hist":
            # pose history is expressed in the body frame of the first pose
            pose_hist = express_in_body_frame(pose_hist, p_t_H, se2_idxs)
            transformed_x[k] = pose_hist
        elif k == "planner_poses" or k == "poses":
            # future poses are expressed in the body frame of the current pose
            transformed_x[k] = express_in_body_frame(v, p_t, se2_idxs)
        else:
            transformed_x[k] = v

    return transformed_x


def _world_frame_poses(
    x: types.ModelInputs, y: types.ModelOutputs, se2_idxs: dict[str, int]
) -> types.ModelInputs:
    """Transforms the poses to the world frame"""
    pose_hist: jnp.ndarray = x["pose_hist"]
    p_t = pose_hist[:, -1]

    transformed_y = {}
    for k, v in y.items():
        if k == "poses":
            # future poses are expressed in the world frame of the current pose
            transformed_y[k] = express_in_world_frame(v, p_t, se2_idxs)
        else:
            transformed_y[k] = v

    return transformed_y


def get_model(
    cfg: dict, scaler_params: types.ScalerParams, rngs: nnx.Rngs
) -> ModelBase:
    models = {
        "world_model": WorldModel,
        "behavior_cloning_with_pose_hist": BehaviorCloningWithPoseHist,
    }
    model_type = cfg["model"]["type"]
    if model_type not in models:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    return models[model_type](cfg, scaler_params, rngs)

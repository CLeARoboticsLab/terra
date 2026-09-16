import terra.modeling.modules as modules
import terra.utils.config as config_utils
from terra.modeling import types
import flax.nnx as nnx
import jax
import jax.numpy as jnp


class HistoryEmbedding(nnx.Module):
    def __init__(self, cfg: dict, rngs: nnx.Rngs, embed_poses: bool = True):
        self.embed_poses = embed_poses
        model_cfg = cfg["model"]
        sys_cfg = cfg["system"]
        self.embed_dim = model_cfg["embed_dim"]
        H = model_cfg["dataset"]["history_length"]
        self.hist_emb_len = 3 * H + 2 if self.embed_poses else 2 * H + 1
        self.obs_dim = config_utils.obs_dim_from_sys_config(sys_cfg)
        self.poses_dim = config_utils.pose_dim_from_sys_config(sys_cfg)

        self.h_size = self.embed_dim * model_cfg["embedding"]["mlp_hidden_size_factor"]
        self.obs_enc = modules.MLP(
            input_dim=self.obs_dim,
            hidden_dims=[self.h_size] * model_cfg["embedding"]["obs_layers"],
            output_dim=self.embed_dim,
            rngs=rngs,
        )
        self.act_enc = modules.MLP(
            input_dim=sys_cfg["acts"]["dim"],
            hidden_dims=[self.h_size] * model_cfg["embedding"]["acts_layers"],
            output_dim=self.embed_dim,
            rngs=rngs,
        )
        if self.embed_poses:
            self.pose_enc = modules.MLP(
                input_dim=self.poses_dim,
                hidden_dims=[self.h_size] * model_cfg["embedding"]["poses_hist_layers"],
                output_dim=self.embed_dim,
                rngs=rngs,
            )

        self.temporal_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (H + 1, self.embed_dim)) * 0.02
        )

        num_types = 3 if self.embed_poses else 2  # obs, act, (poses)
        self.type_enc = nnx.Param(
            jax.random.normal(rngs["params"](), (num_types, self.embed_dim)) * 0.02
        )

    def __call__(
        self,
        x: types.ModelInputs,
        deterministic: bool | None = None,
    ):
        B = x["obs_hist"].shape[0]

        # encode inputs
        obs_enc, acts_enc, poses_enc = self._encode_inputs(
            x, deterministic=deterministic
        )

        # temporal encoding
        obs_emb = obs_enc + self.temporal_enc
        acts_emb = acts_enc + self.temporal_enc[:-1]
        if self.embed_poses:
            poses_emb = poses_enc + self.temporal_enc

        # type encoding
        obs_emb += self.type_enc[0]
        acts_emb += self.type_enc[1]
        if self.embed_poses:
            poses_emb += self.type_enc[2]

        # interleave embeddings
        hist_emb = jnp.zeros((B, self.hist_emb_len, self.embed_dim))
        if self.embed_poses:
            hist_emb = hist_emb.at[:, 0::3].set(obs_emb)
            hist_emb = hist_emb.at[:, 1::3].set(poses_emb)
            hist_emb = hist_emb.at[:, 2::3].set(acts_emb)
        else:
            hist_emb = hist_emb.at[:, 0::2].set(obs_emb)
            hist_emb = hist_emb.at[:, 1::2].set(acts_emb)

        return hist_emb

    def _encode_inputs(self, x: types.ModelInputs, deterministic: bool):
        obs_enc = self.obs_enc(x["obs_hist"], deterministic=deterministic)
        acts_enc = self.act_enc(x["acts_hist"], deterministic=deterministic)
        if self.embed_poses:
            poses_enc = self.pose_enc(x["pose_hist"], deterministic=deterministic)
        else:
            poses_enc = None
        return obs_enc, acts_enc, poses_enc


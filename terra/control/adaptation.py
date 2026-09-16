from terra.modeling import models
from terra.modeling.lora import add_lora_to_transformer
from terra.utils.model import count_params
from terra.modeling.losses import Loss
from terra.modeling import types
from terra.control.rewards import RewardFunctionBase
import flax.nnx as nnx
import jax.numpy as jnp
import jax
import optax
import functools
import operator
import os
import orbax.checkpoint as ocp
from copy import deepcopy
from typing import Sequence, Optional


ADAPT_PARAMS_DIR = "adapt_params"


class AdaptationWrapper(nnx.Module):
    def __init__(
        self,
        model: models.ModelBase,
        loss: Loss,
        reward_fn: RewardFunctionBase,
        learning_rate: float = 1e-3,
        lora: bool = True,
        seed: int = 0,
    ):
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate
        self.lora = lora
        self.norm_eps = 1e-9
        self.train_key = jax.random.PRNGKey(seed)

        self.filt = nnx.LoRAParam if self.lora else nnx.All(nnx.PathContains("dynamics"), nnx.Param)
        loss_pred_filt = nnx.All(nnx.PathContains("loss_pred"), nnx.Param)
        self.filt = nnx.Any(self.filt, loss_pred_filt)
        print(f"Parameters to be adapted: {count_params(model, self.filt)}")

        self.graph_def, self.static_params, self._adapt_params = nnx.split(
            self.model, nnx.Not(self.filt), self.filt
        )
        self._save_last_adapt_params()

        self.tx = optax.adam(self.learning_rate)
        self.opt_state = self.tx.init(self.adapt_params)

        self.lora_paths = ()
        if self.lora:
            self.lora_paths = get_lora_paths(self.adapt_params)

        def reward_only_fn(*args, **kwargs):
            reward = reward_fn(*args, **kwargs)[0]
            return reward

        reward_grad_fn = jax.grad(reward_only_fn, argnums=[3, 4])  # next_pose, next_obs
        reward_grad_fn = jax.vmap(  # over B
            jax.vmap(reward_grad_fn),  # over T
        )

        def loss_fn(
            adapt_params: nnx.GraphState,
            x: types.ModelInputs,
            y: types.ModelOutputs,
            rng_key: jax.Array,
        ):
            kwargs = {}
            model = nnx.merge(self.graph_def, self.static_params, adapt_params)

            reward_in = reward_fn.model_io_to_reward_fn_inputs(x["path_line_def"], x, y)
            grad_reward_wrt_pose, grad_reward_wrt_obs = reward_grad_fn(*reward_in)
            kwargs["grad_reward_wrt_pose"] = grad_reward_wrt_pose
            kwargs["grad_reward_wrt_obs"] = grad_reward_wrt_obs

            stats = (
                lora_deltaW_stats(self.lora_paths, adapt_params, eps=self.norm_eps)
                if self.lora
                else {}
            )

            if self.lora:
                kwargs["norm_deltaW"] = stats["global_norm_sq"]
            kwargs["rng_key"] = rng_key

            return self.loss(model, x, y, False, **kwargs)

        def _train_step(
            adapt_params: nnx.GraphState,
            opt_state: optax.OptState,
            x: types.ModelInputs,
            y: types.ModelOutputs,
            rng_key: jax.Array,
            apply_update: bool,
        ):
            rng_key, loss_key = jax.random.split(rng_key)
            grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
            (loss, losses), grads = grad_fn(adapt_params, x, y, loss_key)

            # 1. Check if the loss is finite
            loss_finite = jnp.isfinite(loss)

            # 2. Check if all gradient leaves are finite
            # jnp.isfinite returns an array; .all() reduces it to a scalar bool
            grads_finite = jax.tree_util.tree_reduce(
                lambda x, y: x & y,
                jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads),
            )

            # 3. Combine conditions
            should_update = apply_update & loss_finite & grads_finite
            nan_detected = ~(loss_finite & grads_finite)

            def _do_update(opt_state, grads, adapt_params):
                updates, new_opt_state = self.tx.update(grads, opt_state, adapt_params)
                new_adapt_params = optax.apply_updates(adapt_params, updates)
                return new_opt_state, new_adapt_params

            def _skip_update(opt_state, grads, adapt_params):
                return opt_state, adapt_params

            new_opt_state, new_adapt_params = nnx.cond(
                should_update,
                _do_update,
                _skip_update,
                opt_state,
                grads,
                adapt_params,
            )

            return loss, losses, new_opt_state, new_adapt_params, rng_key, nan_detected

        self.train_step_fn = nnx.jit(_train_step)

    def __call__(
        self,
        x: types.ModelInputs,
        adapt_params: nnx.GraphState = None,
        deterministic: bool | None = None,
    ) -> types.ModelOutputs:
        if adapt_params is None:
            raise ValueError("adapt_params must be provided to call the adapted model.")
        model = nnx.merge(self.graph_def, self.static_params, adapt_params)
        return model(x, deterministic=deterministic)

    def inference(
        self,
        x: types.ModelInputs,
        adapt_params: nnx.GraphState = None,
        **kwargs,
    ) -> types.ModelOutputs:
        if adapt_params is None:
            raise ValueError("adapt_params must be provided to call the adapted model.")
        model = nnx.merge(self.graph_def, self.static_params, adapt_params)
        return model.inference(x, **kwargs)

    def train_step(
        self,
        x: types.ModelInputs,
        y: types.ModelOutputs,
        apply_update: bool = True,
        compute_metrics: bool = False,
    ):
        (
            _,
            losses,
            self.opt_state,
            self._adapt_params,
            self.train_key,
            was_nan,
        ) = self.train_step_fn(
            self.adapt_params,
            self.opt_state,
            x,
            y,
            self.train_key,
            apply_update,
        )

        if was_nan:
            print(
                "Warning: NaN detected in loss or gradients. Adaptation step skipped.",
                flush=True,
            )

        metrics = self.compute_metrics(losses) if compute_metrics else {}
        return metrics

    def compute_metrics(self, losses: dict):
        losses = jax.tree.map(lambda x: float(x), losses)
        metrics = {"losses": losses}

        if self.lora:
            stats = lora_deltaW_stats(self.lora_paths, self.adapt_params)
            metrics["lora_dw_norm"] = float(stats["global_norm"])

        return metrics

    def save_adapt_params(self, path: Optional[str] = None):
        self._save_last_adapt_params()
        if path is None or path == "":
            return
        ckptr = ocp.PyTreeCheckpointer()
        ckptr.save(
            os.path.join(path, ADAPT_PARAMS_DIR), self.adapt_params.to_pure_dict()
        )

    def load_adapt_params(self, path: Optional[str] = None):
        if path is None or path == "":
            self._adapt_params = self._last_saved_adapt_params
        else:
            ckptr = ocp.PyTreeCheckpointer()
            pure_dict = ckptr.restore(os.path.join(path, ADAPT_PARAMS_DIR))
            self._adapt_params.replace_by_pure_dict(pure_dict)

    def _save_last_adapt_params(self):
        self._last_saved_adapt_params = deepcopy(self._adapt_params)

    @property
    def adapt_params(self):
        return self._adapt_params


class WorldModelAdaptationWrapper(AdaptationWrapper):
    def __init__(
        self,
        model: models.WorldModel,
        model_cfg: dict,
        loss: Loss,
        reward_fn: RewardFunctionBase,
        learning_rate: float = 1e-3,
        lora: bool = True,
        lora_rank: int = 8,
        lora_add_to: Sequence[str] = ("mlp",),
        seed: int = 0,
    ):
        if lora:
            add_lora_to_transformer(
                model.dynamics,
                nnx.Rngs(seed),
                add_to=lora_add_to,
                lora_rank=lora_rank,
            )
        super().__init__(model, loss, reward_fn, learning_rate, lora, seed=seed)


def get_lora_paths(adapt_params: nnx.GraphState):
    buckets = {}
    for path, p in adapt_params.flat_state():
        leaf = str(path[-1])
        if leaf not in ("lora_a", "lora_b"):
            continue
        parent = tuple(path[:-1])
        buckets.setdefault(parent, {})[leaf] = path
    pairs = []
    for parent, d in sorted(buckets.items()):
        if "lora_a" in d and "lora_b" in d:
            pairs.append((tuple(d["lora_a"]), tuple(d["lora_b"])))
    return tuple(pairs)  # static structure


def _getitem_path(obj, path):
    """Helper to index nested mapping by a tuple path: obj[a][b][c]..."""
    return functools.reduce(operator.getitem, path, obj)


@functools.partial(jax.jit, static_argnames=["lora_paths"])
def lora_deltaW_stats(lora_paths, adapt_params: nnx.GraphState, eps: float = 0.0):
    """Compute Frobenius norm of ΔW = A @ B per LoRA block."""
    total_sq = 0.0
    for path_a, path_b in lora_paths:
        A = _getitem_path(adapt_params, path_a).value
        B = _getitem_path(adapt_params, path_b).value
        dW = A @ B
        total_sq += jnp.sum(dW * dW)
    return {"global_norm": jnp.sqrt(total_sq + eps), "global_norm_sq": total_sq}

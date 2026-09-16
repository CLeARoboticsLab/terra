from terra.modeling.models import (
    ModelBase,
    WorldModelWithLossPredTransformer,
)
from terra.control.controller import Controller, ControllerBufferItem
from terra.control.rewards import RewardFunctionBase
from terra.control import sensor_noise as sensor_noise_utils
from terra.modeling.types import ModelInputs, ModelOutputs
from terra.utils import model as model_utils
from terra.utils import config as config_utils
from terra.utils.buffer import MultiRingBuffer
from terra.control.adaptation import WorldModelAdaptationWrapper
from terra.modeling.losses import get_loss
import jax
import numpy as np
import flax
from flax import nnx
import jax.numpy as jnp
import functools
import numbers
from typing import TypedDict, Optional


# Index of the roll angle within a distilled observation vector, and the default
# magnitude limit (degrees) used when the constraint is enabled without one.
ROLL_OBS_IDX = 0
DEFAULT_ROLL_LIMIT_DEG = 15.0


@flax.struct.dataclass
class MppiState:
    mean: jnp.ndarray
    std: jnp.ndarray
    key_data: jnp.ndarray


class MppiControllerOutputs(TypedDict):
    actions: jnp.ndarray  # actions to be applied (act_dim,)
    std: jnp.ndarray  # action stds at final mppi iteration (T, act_dim)
    predicted_poses: jnp.ndarray  # predicted poses (T, pose_dim)


class MppiController(Controller):
    def __init__(
        self,
        model: ModelBase,
        model_cfg: dict,
        world_model: ModelBase,
        world_model_cfg: dict,
        mppi_cfg: dict,
        reward_fn: RewardFunctionBase,
        *args,
        sensor_noise_cfg: Optional[dict] = None,
        sensor_noise_scaler_stds: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(model, model_cfg, *args, **kwargs)
        self.world_model = world_model
        self.world_model_cfg = world_model_cfg
        self.mppi_cfg = mppi_cfg
        self.sensor_noise_cfg = sensor_noise_cfg
        self.sensor_noise = self._make_sensor_noise_injector(
            sensor_noise_cfg, world_model, sensor_noise_scaler_stds
        )
        # Private two-entry history of *clean* measurements, used only to report
        # the actual reward while planning/adaptation stay noise-conditioned.
        # Allocated lazily, and only when sensor noise is enabled.
        self.clean_reward_buf = None
        self.reward_fn = reward_fn
        self.reward_fn_jit = jax.jit(reward_fn)

        self.B = mppi_cfg.get("num_samples", 1000)
        self.min_std = mppi_cfg.get("min_std", 0.001)
        init_std = mppi_cfg.get("init_std", 1.0)
        self.init_std = jnp.array(init_std)
        action_min = mppi_cfg.get("action_min", -1.0)
        self.action_min = jnp.array(action_min)
        action_max = mppi_cfg.get("action_max", 1.0)
        self.action_max = jnp.array(action_max)
        bc_act_std = mppi_cfg.get("bc_act_std", 0.025)
        self.bc_act_std = jnp.array(bc_act_std)
        self.num_elites = mppi_cfg.get("num_elites", 100)
        self.temperature = mppi_cfg.get("temperature", 0.25)
        self.momentum = mppi_cfg.get("momentum", 0.0)
        self.discount = mppi_cfg.get("discount", 0.99)
        self.mixture_coef = mppi_cfg.get("mixture_coef", 0.05)
        self.J = mppi_cfg.get("iterations", 6)
        self.num_bc_trajs = int(self.B * self.mixture_coef)
        assert self.num_bc_trajs > 0
        self.action_noise = mppi_cfg.get("action_noise", "white")
        if self.action_noise == "white":
            self._action_noise_fn = self._white_noise
        elif self.action_noise == "brownian":
            self._action_noise_fn = self._brownian_noise
        else:
            raise ValueError(f"Unknown action noise type: {self.action_noise}")
        self.exploration_reward_weight = mppi_cfg.get("exploration_reward_weight", 0.1)

        self._configure_roll_limit(mppi_cfg)

        self.mppi_state = None
        self.shift = 1
        self.discounts = self.discount ** jnp.arange(self.T)

    def reset(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        if self.sensor_noise is None:
            super().reset(state, system_pose, act)
        else:
            # One noisy realization fills every slot of the control history, so
            # the reset boundary looks like a single repeated measurement.
            with self.buf_lock:
                clean_item, noisy_item = self._ingest_measurement(
                    state, system_pose, act
                )
                self._fill_buffers(noisy_item)
                self._fill_clean_reward_buf(clean_item)
        self.mppi_state = MppiState(
            mean=self._to_dev_f32(jnp.zeros((self.T, self.act_dim))),
            std=self._to_dev_f32(jnp.ones((self.T, self.act_dim)) * self.init_std),
            key_data=self._to_key_data(self.rngs()),
        )

    def update(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        """Updates the controller's buffer; should be called every dt."""
        if self.sensor_noise is None:
            super().update(state, system_pose, act)
            return
        with self.buf_lock:
            clean_item, noisy_item = self._ingest_measurement(state, system_pose, act)
            self.buf.append(noisy_item)
            self.clean_reward_buf.append(clean_item)

    def _make_sensor_noise_injector(
        self,
        sensor_noise_cfg: Optional[dict],
        world_model: ModelBase,
        scaler_stds: Optional[dict],
    ) -> Optional[sensor_noise_utils.SensorNoiseInjector]:
        """Builds the sensor-noise injector, or None when the feature is off.

        `scaler_stds` lets subclasses hand in standard deviations captured from
        the base world model before it was wrapped for online adaptation.
        """
        if not sensor_noise_utils.sensor_noise_enabled(sensor_noise_cfg):
            return None
        if scaler_stds is None:
            scaler_stds = sensor_noise_utils.scaler_stds_from_model(world_model)
        return sensor_noise_utils.make_sensor_noise_injector(
            sensor_noise_cfg,
            scaler_stds,
            obs_dim=config_utils.obs_dim_from_sys_config(self.sys_cfg),
            pose_dim=self.pose_dim,
            yaw_idx=self.sys_cfg["poses"]["se2_idxs"]["yaw"],
        )

    def _ingest_measurement(
        self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray
    ) -> tuple[ControllerBufferItem, ControllerBufferItem]:
        """Distills one incoming measurement into a clean and a noisy item.

        Exactly one noise realization is drawn, so all consumers of the noisy
        item see the same corruption. Callers must hold `buf_lock` so that
        noise sampling stays serialized with buffer ingestion.
        """
        clean_item = self._buf_item(state, system_pose, act)
        return clean_item, self.sensor_noise.perturb(clean_item)

    def _fill_buffers(self, fill_item: ControllerBufferItem):
        """Fills every measurement buffer from one already-built noisy item."""
        self.buf.fill(fill_item)

    def _fill_clean_reward_buf(self, clean_item: ControllerBufferItem):
        """Fills the clean reward history from one clean item.

        The buffer is created here on first use; building the schema consumes
        no sensor-noise draw. Refilling on every reset keeps the reported
        reward transition inside the new episode.
        """
        if self.clean_reward_buf is None:
            self.clean_reward_buf = MultiRingBuffer(maxlen=2, example_item=clean_item)
        self.clean_reward_buf.fill(clean_item)

    def run_control(self, planner_poses: np.ndarray) -> MppiControllerOutputs:
        # Run the base controller to get BC actions
        bc_actions = self._get_bc_actions(planner_poses)

        # Make the model inputs from the buffer
        model_inputs = self._make_world_model_inputs()

        # Step the MPPI algorithm
        self.mppi_state = self._mppi_step(
            self.mppi_state,
            bc_actions,
            model_inputs,
            self.path_line_def,
        )
        fut_actions = self.mppi_state.mean
        fut_std = self.mppi_state.std

        # With the given future actions, predict the future poses
        traj = self._predict_future_trajectory(
            model_inputs, fut_actions, self.path_line_def
        )

        return {
            "actions": np.array(fut_actions),
            "std": np.array(fut_std),
            **jax.tree.map(self._to_scalar, traj),
        }

    def compute_current_reward(self):
        model_inputs, model_outputs = self._make_model_io_for_reward_fn()
        reward, reward_comps, reward_comps_unweighted = self._calc_reward(
            self.path_line_def, model_inputs, model_outputs
        )

        reward_data = {
            "reward": reward,
            "reward_comps": reward_comps,
            "reward_comps_unweighted": reward_comps_unweighted,
        }

        return jax.tree.map(self._to_scalar, reward_data)

    def _get_bc_actions(self, planner_poses: np.ndarray) -> jnp.ndarray:
        control_out = super().run_control(planner_poses)
        bc_actions = self._to_dev_f32(control_out["actions"])
        return bc_actions  # (T, act_dim)

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _mppi_step(
        self,
        mppi_state: MppiState,
        bc_actions: jnp.ndarray,
        model_inputs: ModelInputs,
        path_def: jnp.ndarray,
        **kwargs,
    ) -> MppiState:
        key = self._to_key(mppi_state.key_data)

        # add noise to BC actions
        key, key_bc_act = jax.random.split(key)
        bc_act_noise = self._action_noise_fn(
            key_bc_act, self.num_bc_trajs, self.bc_act_std
        )
        noised_bc_acts = bc_actions + bc_act_noise

        # initialize
        prev_mean = mppi_state.mean
        mean = jnp.roll(prev_mean, shift=self.shift, axis=0)
        mean = mean.at[self.shift :].set(0.0)
        std = jnp.ones((self.T, self.act_dim)) * self.init_std

        def f(carry, xs_unused):
            mean, std, key = carry

            # get noised actions
            key, key_act = jax.random.split(key)
            act_noise = self._action_noise_fn(key_act, self.B, std)
            noised_acts = mean + act_noise

            # run model with both BC and noised actions
            acts = jnp.concatenate([noised_acts, noised_bc_acts], axis=0)
            acts = jnp.clip(acts, self.action_min, self.action_max)
            world_model_inputs = model_utils.repeat_along_batch_dim(
                model_inputs, acts.shape[0]
            )
            world_model_inputs["fut_acts"] = acts
            model_outputs = self.world_model.inference(world_model_inputs, **kwargs)
            rewards = self._calc_reward(path_def, world_model_inputs, model_outputs)[0]

            if not self.roll_limit_enabled:
                mean, std = self._update_action_distribution(rewards, acts, mean, std)
                return (mean, std, key), None

            # Planner-local selection returns only: the rewards above (and every
            # reported reward quantity) stay untouched by the constraint.
            feasible = self._roll_feasible_mask(model_outputs["obs"])
            selection_returns = jnp.where(feasible, rewards, -jnp.inf)
            # With zero feasible candidates every elite weight would be
            # exp(-inf - -inf) = NaN, so that iteration keeps its incoming
            # distribution instead and the next one resamples from it.
            mean, std = jax.lax.cond(
                jnp.any(feasible),
                lambda m, s: self._update_action_distribution(
                    selection_returns, acts, m, s
                ),
                lambda m, s: (m, s),
                mean,
                std,
            )

            return (mean, std, key), None

        (mean, std, key), _ = jax.lax.scan(f, (mean, std, key), jnp.arange(self.J))

        return MppiState(
            self._to_dev_f32(mean),
            self._to_dev_f32(std),
            self._to_key_data(key),
        )

    def _configure_roll_limit(self, mppi_cfg: dict):
        """Reads the optional hard roll-angle constraint on planner candidates.

        The config boundary is degrees while predicted roll is denormalized
        radians, so the validated limit is converted once here. When the feature
        is disabled nothing else about it is read or validated, and planning
        traces no feasibility operations at all.
        """
        self.roll_limit_enabled = _validate_roll_limit_enabled(
            mppi_cfg.get("roll_limit_enabled", False)
        )
        self.roll_limit_rad = None
        if self.roll_limit_enabled:
            roll_limit_deg = _validate_roll_limit_deg(
                mppi_cfg.get("roll_limit_deg", DEFAULT_ROLL_LIMIT_DEG)
            )
            self.roll_limit_rad = float(np.deg2rad(roll_limit_deg))

    def _update_action_distribution(
        self,
        returns: jnp.ndarray,
        acts: jnp.ndarray,
        mean: jnp.ndarray,
        std: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Selects elites by `returns` and refits the sampling distribution.

        `returns` is whatever the planner selects on: the ordinary rewards, or
        the roll-masked selection returns in which infeasible candidates carry
        `-inf` and therefore receive zero exponential weight.
        """
        # select elites
        _, elite_idxs = jax.lax.top_k(returns, self.num_elites)
        elite_rews = jnp.take(returns, elite_idxs, axis=0)
        elite_acts = jnp.take(acts, elite_idxs, axis=0)

        # update
        max_rew = jnp.max(elite_rews)
        score = jnp.exp(self.temperature * (elite_rews - max_rew))
        _mean = jnp.sum(score[:, None, None] * elite_acts, axis=0) / jnp.sum(score)
        _std = jnp.sqrt(
            jnp.sum(score[:, None, None] * (elite_acts - _mean) ** 2, axis=0)
            / jnp.sum(score)
        )

        mean = self.momentum * mean + (1 - self.momentum) * _mean
        std = jnp.clip(_std, min=self.min_std)

        return mean, std

    def _roll_feasible_mask(self, pred_obs: jnp.ndarray) -> jnp.ndarray:
        """Returns a (candidate,) bool mask of roll-feasible candidates.

        `pred_obs` is the denormalized predicted observation tensor of shape
        (candidate, time, obs_dim). A candidate is feasible only if the roll of
        every predicted future step is finite and within the symmetric limit;
        the current measured observation is deliberately not part of the check,
        so a rover already past the limit can still plan a recovery.
        """
        roll = pred_obs[..., ROLL_OBS_IDX]
        step_feasible = jnp.isfinite(roll) & (jnp.abs(roll) <= self.roll_limit_rad)
        return jnp.all(step_feasible, axis=-1)

    def _make_world_model_inputs(self) -> ModelInputs:
        with self.buf_lock:
            hist = self.buf.get()
            model_inputs = {
                "obs_hist": hist["obs"][-self.obs_hist_len :],
                "pose_hist": hist["pose"][-self.pose_hist_len :],
                "acts_hist": hist["act"][-self.act_hist_len :],
            }
            model_inputs = jax.tree.map(self._to_dev_f32, model_inputs)
        return model_inputs

    def _make_model_io_for_reward_fn(self) -> tuple[ModelInputs, ModelOutputs]:
        with self.buf_lock:
            hist = self._reward_hist()
            model_inputs = {
                "obs_hist": hist["obs"][-2:-1],
                "pose_hist": hist["pose"][-2:-1],
                "acts_hist": hist["act"][-2:-1],
                "fut_acts": hist["act"][-1:],
            }
            model_outputs = {
                "obs": hist["obs"][-1:],
                "poses": hist["pose"][-1:],
            }
        model_inputs = jax.tree.map(self._to_dev_f32, model_inputs)
        model_inputs = model_utils.repeat_along_batch_dim(model_inputs, 1)
        model_outputs = jax.tree.map(self._to_dev_f32, model_outputs)
        model_outputs = model_utils.repeat_along_batch_dim(model_outputs, 1)

        return model_inputs, model_outputs

    def _reward_hist(self) -> dict[str, np.ndarray]:
        """Returns the measurement history the actual reward is computed from.

        Without sensor noise this is the main buffer, exactly as before. With
        sensor noise it is the private clean history, so the reported reward
        reflects what actually happened rather than the corrupted measurements
        that planning and adaptation are deliberately conditioned on. Callers
        must hold `buf_lock`.
        """
        if self.sensor_noise is None:
            return self.buf.get()
        if self.clean_reward_buf is None:
            raise RuntimeError(
                "Sensor noise is enabled but the clean reward history has not "
                "been created; the controller must be initialized before "
                "computing the actual reward."
            )
        if self.clean_reward_buf.length() < 2:
            raise RuntimeError(
                "Sensor noise is enabled but the clean reward history holds "
                f"{self.clean_reward_buf.length()} of the 2 measurements needed "
                "for an actual-reward transition."
            )
        return self.clean_reward_buf.get()

    def _calc_reward(
        self,
        path_def: jnp.ndarray,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        reward_in = self.reward_fn.model_io_to_reward_fn_inputs(
            path_def, model_inputs, model_outputs
        )
        reward, reward_comps, reward_comps_unweighted = self.reward_fn_jit(*reward_in)

        if "exploration_reward" in model_outputs:
            exploration_reward = model_outputs["exploration_reward"]
            reward += self.exploration_reward_weight * exploration_reward
            reward_comps["exploration_reward"] = (
                self.exploration_reward_weight * exploration_reward
            )
            reward_comps_unweighted["exploration_reward"] = exploration_reward

        # apply discounting
        T = reward.shape[1]
        reward = reward * self.discounts[None, :T]
        reward_comps = jax.tree.map(
            lambda x: x * self.discounts[None, :T], reward_comps
        )
        reward_comps_unweighted = jax.tree.map(
            lambda x: x * self.discounts[None, :T], reward_comps_unweighted
        )

        # sum over time
        reward = jnp.sum(reward, axis=1)
        reward_comps = jax.tree.map(lambda x: jnp.sum(x, axis=1), reward_comps)
        reward_comps_unweighted = jax.tree.map(
            lambda x: jnp.sum(x, axis=1), reward_comps_unweighted
        )

        return reward, reward_comps, reward_comps_unweighted

    @functools.partial(nnx.jit, static_argnames=["self"])
    def _predict_future_trajectory(
        self,
        model_inputs: ModelInputs,
        fut_actions: jnp.ndarray,
        path_def: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """Predict future poses given the model inputs and future actions."""
        # Prepare inputs for the world model
        model_inputs["fut_acts"] = fut_actions
        world_model_inputs = model_utils.repeat_along_batch_dim(model_inputs, 1)

        model_outputs = self.world_model.inference(world_model_inputs, **kwargs)
        reward, reward_comps, reward_comps_unweighted = self._calc_reward(
            path_def, world_model_inputs, model_outputs
        )

        return {
            "predicted_poses": model_outputs["poses"][0],
            "reward": reward[0],
            "reward_comps": jax.tree.map(lambda x: x[0], reward_comps),
            "reward_comps_unweighted": jax.tree.map(
                lambda x: x[0], reward_comps_unweighted
            ),
        }

    def _init_fn(self, dummy_planner_poses: np.ndarray):
        self.run_control(dummy_planner_poses)
        return self.run_control(dummy_planner_poses)

    def _white_noise(self, key: jnp.ndarray, batch_size: int, std: jnp.ndarray):
        return jax.random.normal(key, (batch_size, self.T, self.act_dim)) * std

    def _brownian_noise(self, key: jnp.ndarray, batch_size: int, std: jnp.ndarray):
        eps = jax.random.normal(key, (batch_size, self.T, self.act_dim)) * std
        noise = jnp.cumsum(eps, axis=1)
        return noise

    def _to_scalar(self, x: jnp.ndarray):
        x_np = np.asarray(x)
        squeezed = x_np.squeeze()
        if squeezed.shape == ():
            return squeezed.item()
        return x_np


class MppiControllerWithAdaptation(MppiController):
    def __init__(
        self,
        model: ModelBase,
        model_cfg: dict,
        world_model: ModelBase,
        world_model_cfg: dict,
        mppi_cfg: dict,
        reward_fn: RewardFunctionBase,
        adaptation_cfg: dict,
        *args,
        sensor_noise_cfg: Optional[dict] = None,
        **kwargs,
    ):
        # Capture the dataset statistics from the pretrained world model
        # before it is wrapped for exploration/online adaptation.
        sensor_noise_scaler_stds = None
        if sensor_noise_utils.sensor_noise_enabled(sensor_noise_cfg):
            sensor_noise_scaler_stds = sensor_noise_utils.scaler_stds_from_model(
                world_model
            )

        self.H = model_cfg["model"]["dataset"]["history_length"]
        self.T = model_cfg["model"]["dataset"]["prediction_length"]
        self.adapt_buf_len = adaptation_cfg["buffer_length"]
        assert self.adapt_buf_len >= self.H + self.T + 1

        self.batch_size = adaptation_cfg["batch_size"]
        self.control_steps_before_adapt = adaptation_cfg["control_steps_before_adapt"]
        self.always_include_latest = adaptation_cfg["always_include_latest"]
        self.seq_len = self.H + self.T + 1
        seed = adaptation_cfg.get("seed", 0)

        # State
        self.control_steps = 0
        self.is_very_first_update = True
        self.was_reset = False
        # True only while initialize() is filling the buffers, so that the
        # adaptation buffer shares the control buffer's first measurement.
        self._pending_adapt_fill = False

        # Override world model loss with the one from adaptation config in case the
        # values or terms are different
        assert "loss" in adaptation_cfg, "Adaptation config must specify a loss."
        world_model_cfg["loss"] = adaptation_cfg["loss"]

        use_loss_pred = adaptation_cfg.get("loss_pred", {}).get("use_loss_pred", False)

        if use_loss_pred:
            print("Using loss predictor for adaptation.", flush=True)
            world_model_cfg["loss_pred"] = adaptation_cfg["loss_pred"]

            world_model = WorldModelWithLossPredTransformer(
                world_model, world_model_cfg, rngs=nnx.Rngs(seed)
            )
            world_model_cfg["loss"]["type"] = "world_model_adaptation_loss_pred"
            world_model_cfg["loss"]["weights"]["loss_pred_loss"] = adaptation_cfg[
                "loss_pred"
            ]["loss_pred_loss_weight"]
            world_model_cfg["loss"]["loss_pred_thresholds"] = adaptation_cfg[
                "loss_pred"
            ]["thresholds"]
            if "class_weights" in adaptation_cfg["loss_pred"]:
                class_weights = adaptation_cfg["loss_pred"]["class_weights"]
                if not _is_none_str(class_weights):
                    world_model_cfg["loss"]["loss_pred_class_weights"] = class_weights
        use_lora = adaptation_cfg.get("lora", True)
        if not use_lora:
            if "norm_deltaW_loss" in adaptation_cfg["loss"]["weights"]:
                print(
                    "Warning: LoRA is disabled but norm_deltaW_loss_weight was given. "
                    "Removing norm_deltaW_loss_weight"
                )
                adaptation_cfg["loss"]["weights"].pop("norm_deltaW_loss")

        loss = get_loss(world_model_cfg)

        world_model = WorldModelAdaptationWrapper(
            model=world_model,
            model_cfg=world_model_cfg,
            loss=loss,
            reward_fn=reward_fn,
            learning_rate=adaptation_cfg.get("learning_rate", 1e-3),
            lora=use_lora,
            lora_rank=adaptation_cfg.get("lora_rank", 8),
            lora_add_to=adaptation_cfg.get("lora_add_to", ("mlp",)),
            seed=seed,
        )

        super().__init__(
            model=model,
            model_cfg=model_cfg,
            world_model=world_model,
            world_model_cfg=world_model_cfg,
            mppi_cfg=mppi_cfg,
            reward_fn=reward_fn,
            sensor_noise_cfg=sensor_noise_cfg,
            sensor_noise_scaler_stds=sensor_noise_scaler_stds,
        )
        self.world_model: WorldModelAdaptationWrapper = self.world_model  # for annotations

    def reset(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        super().reset(state, system_pose, act)
        if not self.is_very_first_update:
            self.was_reset = True

    def initialize(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        """Initializes the controller by filling buffers with the given
        observation (distilled from state), pose, and action, and jitting all
        operations. Must be called before run_control()
        """
        fill_item = self._buf_item(state, system_pose, act)
        self.adapt_buf = MultiRingBuffer(
            maxlen=self.adapt_buf_len, example_item=fill_item
        )
        if self.sensor_noise is None:
            with self.buf_lock:
                self.adapt_buf.fill(fill_item)
            self.is_very_first_update = True
            super().initialize(state, system_pose, act)
            return

        # The example item above only defines shapes/dtypes and consumes no
        # noise draws. Both buffers are then filled from the single noisy
        # measurement sampled by the reset inside ControllerBase.initialize().
        self.is_very_first_update = True
        self._pending_adapt_fill = True
        try:
            super().initialize(state, system_pose, act)
        finally:
            self._pending_adapt_fill = False

    def update(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        """Updates the controller's buffer; should be called every dt."""
        episode_start = self.was_reset and not self.is_very_first_update
        if self.sensor_noise is None:
            item = self._buf_item(state, system_pose, act)
            with self.buf_lock:
                self.adapt_buf.append(item, episode_start=episode_start)
        else:
            # One realization is shared by the planning and adaptation buffers;
            # only the actual-reward history sees the clean measurement.
            with self.buf_lock:
                clean_item, item = self._ingest_measurement(state, system_pose, act)
                self.adapt_buf.append(item, episode_start=episode_start)
                self.buf.append(item)
                self.clean_reward_buf.append(clean_item)
        if self.was_reset:
            self.was_reset = False
        if self.is_very_first_update:
            self.is_very_first_update = False
        if self.sensor_noise is None:
            super().update(state, system_pose, act)

    def _fill_buffers(self, fill_item: ControllerBufferItem):
        super()._fill_buffers(fill_item)
        if self._pending_adapt_fill:
            self.adapt_buf.fill(fill_item)

    def run_control(self, planner_poses: np.ndarray) -> MppiControllerOutputs:
        # Run the base controller to get BC actions
        bc_actions = self._get_bc_actions(planner_poses)

        # Make the model inputs from the buffer
        model_inputs = self._make_world_model_inputs()

        # Step the MPPI algorithm
        self.mppi_state = self._mppi_step(
            self.mppi_state,
            bc_actions,
            model_inputs,
            self.path_line_def,
            adapt_params=self.world_model.adapt_params,
        )
        fut_actions = self.mppi_state.mean
        fut_std = self.mppi_state.std

        # With the given future actions, predict the future poses
        traj = self._predict_future_trajectory(
            model_inputs,
            fut_actions,
            self.path_line_def,
            adapt_params=self.world_model.adapt_params,
        )

        if self.initialized:
            self.control_steps += 1

        return {
            "actions": np.array(fut_actions),
            "std": np.array(fut_std),
            **jax.tree.map(self._to_scalar, traj),
        }

    def adapt(self, compute_metrics: bool = False):
        with self.buf_lock:
            batch = self.adapt_buf.sample_sequences(
                self.batch_size,
                self.seq_len,
                include_latest=self.always_include_latest,
                sample_from_latest=self.control_steps - self.seq_len,
            )
        if batch is None:
            raise ValueError("Not enough data in buffer to do adaptation.")
        x = self._world_model_in_from_batch(batch)
        y = self._world_model_out_from_batch(batch)

        skip_update = (
            self.is_initing or self.control_steps < self.control_steps_before_adapt
        )
        metrics = self.world_model.train_step(
            x, y, apply_update=not skip_update, compute_metrics=compute_metrics
        )
        return metrics

    def save_adaptation(self, path: Optional[str] = None):
        self.world_model.save_adapt_params(path)
        with self.buf_lock:
            self.adapt_buf.save(path)

    def load_adaptation(self, path: Optional[str] = None):
        self.world_model.load_adapt_params(path)
        with self.buf_lock:
            self.adapt_buf.load(path)

    def _init_fn(self, dummy_planner_poses: np.ndarray):
        outputs = super()._init_fn(dummy_planner_poses)
        self.adapt()
        outputs = super()._init_fn(dummy_planner_poses)
        self.adapt()
        return outputs

    def _world_model_in_from_batch(self, batch: dict[str, np.ndarray]) -> ModelInputs:
        model_inputs = {
            "obs_hist": batch["obs"][:, : self.H + 1],
            "pose_hist": batch["pose"][:, : self.H + 1],
            "acts_hist": batch["act"][:, 1 : self.H + 1],  # actions are delayed by one
            "fut_acts": batch["act"][:, self.H + 1 :],
            "path_line_def": batch["path_line_def"][:, self.H + 1 :],
        }
        model_inputs = jax.tree.map(self._to_dev_f32, model_inputs)
        return model_inputs

    def _world_model_out_from_batch(self, batch: dict[str, np.ndarray]) -> ModelOutputs:
        model_outputs = {
            "obs": batch["obs"][:, self.H + 1 :],
            "poses": batch["pose"][:, self.H + 1 :],
            "pose_hist": batch["pose"][:, : self.H + 1],
        }
        model_outputs = jax.tree.map(self._to_dev_f32, model_outputs)
        return model_outputs

    def _buf_item(
        self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray
    ) -> ControllerBufferItem:
        item = super()._buf_item(state, system_pose, act)
        item["path_line_def"] = np.array(self.path_line_def)
        return item


def _is_none_str(value) -> bool:
    return isinstance(value, str) and value.lower() == "none"


def _validate_roll_limit_enabled(enabled) -> bool:
    """Validates the roll-limit enable flag, which defaults to disabled.

    Only a real bool enables the feature; a truthy string such as "false" would
    otherwise silently turn the constraint on.
    """
    if not isinstance(enabled, bool):
        raise ValueError(
            f"mppi.roll_limit_enabled must be a bool, got {enabled!r}."
        )
    return enabled


def _validate_roll_limit_deg(limit_deg) -> float:
    """Validates the roll magnitude limit in degrees, on the enabled path only."""
    if isinstance(limit_deg, bool) or not isinstance(limit_deg, numbers.Real):
        raise ValueError(
            f"mppi.roll_limit_deg must be a real scalar, got {limit_deg!r}."
        )
    limit_deg = float(limit_deg)
    if not np.isfinite(limit_deg):
        raise ValueError(f"mppi.roll_limit_deg must be finite, got {limit_deg}.")
    if limit_deg <= 0.0:
        raise ValueError(
            f"mppi.roll_limit_deg must be strictly positive, got {limit_deg}."
        )
    return limit_deg



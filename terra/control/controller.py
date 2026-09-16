from terra.modeling.models import ModelBase
from terra.modeling.types import ModelInputs
from terra.utils import model as model_utils
from terra.utils import config as config_utils
from terra.utils.buffer import MultiRingBuffer
import numpy as np
import math
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import threading
from typing import TypedDict


class ControllerBufferItem(TypedDict):
    obs: np.ndarray
    pose: np.ndarray
    act: np.ndarray


class ControllerOutputs(TypedDict):
    actions: jnp.ndarray  # actions to be applied (T, act_dim)


class ControllerBase:
    def __init__(self, model: ModelBase, model_cfg: dict, *args, **kwargs):
        self.model = model
        self.model_cfg = model_cfg
        self.sys_cfg = model_cfg["system"]

        self.H = model_cfg["model"]["dataset"]["history_length"]
        self.T = model_cfg["model"]["dataset"]["prediction_length"]
        self.obs_hist_len = self.H + 1
        self.pose_hist_len = self.H + 1
        self.act_hist_len = self.H
        self.pose_dim = config_utils.pose_dim_from_sys_config(self.sys_cfg)
        self.act_dim = config_utils.acts_dim_from_sys_config(self.sys_cfg)
        self.model_type = model_cfg["model"]["type"]
        self.se2_idxs = np.array(list(self.sys_cfg["poses"]["se2_idxs"].values()))

        if "buffer_length" in kwargs:
            self.buffer_length = kwargs["buffer_length"]
        else:
            self.buffer_length = self.H + 1

        self.buf = None
        self.initialized = False
        self.is_initing = False
        self.buf_lock = threading.Lock()
        self.rngs = nnx.Rngs(0)
        self.device = (
            jax.devices("gpu")[0]
            if any(d.platform == "gpu" for d in jax.devices())
            else jax.devices()[0]
        )

    def reset(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        """Resets the controller by filling the buffer with the given
        observation (distilled from state), pose, and action."""
        fill_item = self._buf_item(state, system_pose, act)
        with self.buf_lock:
            self.buf.fill(fill_item)

    def initialize(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        """Initializes the controller by filling buffers with the given
        observation (distilled from state), pose, and action, and jitting all
        operations. Must be called before run_control()
        """
        fill_item = self._buf_item(state, system_pose, act)
        self.buf = MultiRingBuffer(maxlen=self.buffer_length, example_item=fill_item)
        self.reset(state, system_pose, act)

        # initialze the controller
        cls_name = self.__class__.__name__
        print(f"Initializing {cls_name}...", flush=True)
        dummy_planner_poses = self._make_dummy_planner_poses()
        self.is_initing = True
        _ = self._init_fn(dummy_planner_poses)
        self.initialized = True
        self.is_initing = False
        print(f"{cls_name} initialized.", flush=True)

    def update(self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray):
        """Updates the controller's buffer; should be called every dt."""
        item = self._buf_item(state, system_pose, act)
        with self.buf_lock:
            self.buf.append(item)

    def run_control(self, planner_poses: np.ndarray) -> ControllerOutputs:
        """Returns the actions for the given planner poses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _to_dev_f32(self, x):
        return jax.device_put(jnp.asarray(x, jnp.float32), self.device)

    def _to_dev_u32(self, x):
        return jax.device_put(jnp.asarray(x, jnp.uint32), self.device)

    def _to_key_data(self, key):
        return self._to_dev_u32(jax.random.key_data(key))

    def _to_key(self, key_data):
        return jax.random.wrap_key_data(key_data)

    @property
    def prediction_length(self) -> int:
        """Returns the prediction length of the controller."""
        return self.T

    def _init_fn(self, dummy_planner_poses: np.ndarray):
        """Initializes the controller by calling the main control method once.
        Doing this JITs the function so that future calls are fast."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _buf_item(
        self, state: np.ndarray, system_pose: np.ndarray, act: np.ndarray
    ) -> ControllerBufferItem:
        """Creates a buffer item from the given state, system pose, and action."""
        return {
            "obs": model_utils.obs_from_state(state, self.sys_cfg),
            "pose": model_utils.pose_from_system_pose(system_pose, self.sys_cfg),
            "act": act,
        }

    def _make_model_inputs(self, planner_poses: np.ndarray) -> ModelInputs:
        with self.buf_lock:
            hist = self.buf.get()
            model_inputs = {
                "obs_hist": hist["obs"][-self.obs_hist_len :],
                "pose_hist": hist["pose"][-self.pose_hist_len :],
                "acts_hist": hist["act"][-self.act_hist_len :],
                "planner_poses": planner_poses,
            }
            model_inputs = jax.tree.map(jnp.asarray, model_inputs)
        model_inputs = model_utils.repeat_along_batch_dim(model_inputs, 1)
        return model_inputs

    def _make_dummy_planner_poses(self) -> np.ndarray:
        return np.zeros((self.prediction_length, self.pose_dim), dtype=np.float32)


class Controller(ControllerBase):
    def __init__(self, model: ModelBase, model_cfg: dict, *args, **kwargs):
        super().__init__(model, model_cfg, *args, **kwargs)

        self.set_desired_path_line(0.0, 0.0, 0.0)
        self.model_fn = jax.jit(self.model.inference)

    def set_desired_path_line(self, x0: float, y0: float, yaw: float):
        desired_path_a = -math.sin(yaw)
        desired_path_b = math.cos(yaw)
        desired_path_c = -desired_path_a * x0 - desired_path_b * y0
        self.path_line_def = jnp.array(
            [yaw, desired_path_a, desired_path_b, desired_path_c]
        )

    def run_control(self, planner_poses: np.ndarray) -> ControllerOutputs:
        """Returns the actions for the given planner poses."""
        if not self.initialized and not self.is_initing:
            raise RuntimeError("Controller must be initialized before getting actions.")

        model_inputs = self._make_model_inputs(planner_poses)
        model_outputs = self.model_fn(model_inputs)
        actions = model_outputs["actions"][0]

        return {"actions": np.array(actions)}

    def _init_fn(self, dummy_planner_poses: np.ndarray):
        return self.run_control(dummy_planner_poses)

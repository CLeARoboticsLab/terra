from terra.data import get_data_dir
from terra.utils import config as config_utils
from terra.utils import math as math_utils
from terra.data import Names
from terra.utils.io import save_scaler_params
import os
from tqdm import trange
import yaml
import pickle
import numpy as np
import jax.numpy as jnp
import h5py
import json
from typing import TypedDict
import copy
import gc


EXPERT_FILENAME = "expert_experience.pkl"
EVAL_METRICS_FILENAME = "eval_metrics.json"
RL_PARAMS_FILENAME = "rl_parameters.yaml"
REPLAY_BUFFER_FILENAME = "replay_buffer.pkl"
OBS_IDX = 0
ACTS_IDX = 1
POSES_IDX = 5
TERMINATION_IDX = 3


class AllDataListed(TypedDict):
    all_obs: list[np.ndarray]
    all_acts: list[np.ndarray]
    all_poses: list[jnp.ndarray]
    start_idxs: list[np.ndarray]
    ep_lens: list[int]
    start: int


class DataProcessor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sys = cfg["system"]
        self.data_dir = cfg["data_dir"]
        self.process_replay_buffers = cfg.get("process_replay_buffers", False)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' does not exist.")

        self.H = cfg["history_length"]
        self.T = cfg["prediction_length"]
        self.end_padding = cfg["end_padding"]
        if cfg.get("ignore_max_plan_interval", False):
            self.max_plan_interval = 0
        else:
            self.max_plan_interval = self.sys["poses"]["max_plan_interval"]

        self.expert_dir = get_data_dir(
            cfg["output_dir"], self.sys["type"], self.H, self.T, expert_only=True
        )
        os.makedirs(self.expert_dir, exist_ok=True)
        print(f"Output directory for expert data: {self.expert_dir}")
        if self.process_replay_buffers:
            self.replay_buffer_dir = get_data_dir(
                cfg["output_dir"], self.sys["type"], self.H, self.T, expert_only=False
            )
            os.makedirs(self.replay_buffer_dir, exist_ok=True)
            print(f"Output directory for replay buffers: {self.replay_buffer_dir}")
        config_copy_path = os.path.join(
            os.path.dirname(self.expert_dir), "process_config.yaml"
        )
        with open(config_copy_path, "w") as f:
            yaml.dump(self.cfg, f)

        self.obs_dim = config_utils.obs_dim_from_sys_config(self.sys)
        self.acts_dim = self.sys["acts"]["dim"]
        self.pose_dim = config_utils.pose_dim_from_sys_config(self.sys)

    def process(self):
        print("Processing expert experience...")
        expert_data, expert_scaler_params = self._process_expert_experience()
        if self.process_replay_buffers:
            print("Processing replay buffers...")
            self._process_replay_buffers(expert_data, expert_scaler_params)

    def _process_expert_experience(self) -> tuple[AllDataListed, dict]:
        dirs = self._dirs_with_experience(self.data_dir, EXPERT_FILENAME)
        if len(dirs) == 0:
            raise ValueError(
                f"No directories with expert experience found in {self.data_dir}."
            )
        unfilt_eps = len(dirs)
        print(f"Found {len(dirs)} directories with expert experience.")

        start_idxs = []
        all_obs = []
        all_acts = []
        all_poses = []
        ep_lens = []
        start = 0

        for dir in trange(len(dirs), desc="Processing experience"):
            experience = self._get_experience(
                dirs[dir], EXPERT_FILENAME, start, check_eval_metrics=True
            )
            if experience is None:
                continue
            all_obs.append(experience["obs"])
            all_acts.append(experience["acts"])
            all_poses.append(jnp.array(experience["poses"]))
            start_idxs.append(experience["start_idxs"])
            ep_lens.extend(experience["ep_lens"])
            start = experience["start"]

        all_data_listed = {
            "all_obs": all_obs,
            "all_acts": all_acts,
            "all_poses": all_poses,
            "start_idxs": start_idxs,
            "ep_lens": ep_lens,
            "start": start,
        }

        dataset_metrics = {
            "num_unfiltered_episodes": unfilt_eps,
            "num_episodes": len(ep_lens),
            "avg_episode_length": np.mean(ep_lens),
        }

        all_data_listed_copy = copy.deepcopy(all_data_listed)

        scaler_params = self._save_data(
            all_data_listed, self.expert_dir, metrics=dataset_metrics
        )

        return all_data_listed_copy, scaler_params

    def _process_replay_buffers(
        self, expert_data: AllDataListed, expert_scaler_params: dict
    ):
        dirs = self._dirs_with_experience(self.data_dir, REPLAY_BUFFER_FILENAME)
        if len(dirs) == 0:
            raise ValueError(
                f"No directories with replay buffers found in {self.data_dir}."
            )
        print(f"Found {len(dirs)} directories with replay buffers.")

        obs_h5 = _H5Appender(
            os.path.join(self.replay_buffer_dir, Names.OBS_FILE_NAME),
            "data",
            self.obs_dim,
        )
        acts_h5 = _H5Appender(
            os.path.join(self.replay_buffer_dir, Names.ACTIONS_FILE_NAME),
            "data",
            self.acts_dim,
        )
        poses_h5 = _H5Appender(
            os.path.join(self.replay_buffer_dir, Names.POSES_FILE_NAME),
            "data",
            self.pose_dim,
        )
        start_idxs_h5 = _H5Appender(
            os.path.join(self.replay_buffer_dir, Names.START_IDXS_FILE_NAME),
            "data",
            0,
            dtype=np.int64,
        )

        # add expert data first
        obs_h5.append(np.concatenate(expert_data["all_obs"], axis=0))
        acts_h5.append(np.concatenate(expert_data["all_acts"], axis=0))
        poses_h5.append(np.array(jnp.concatenate(expert_data["all_poses"], axis=0)))
        start_idxs_h5.append(np.concatenate(expert_data["start_idxs"], axis=0))

        ep_lens = expert_data["ep_lens"]
        start = expert_data["start"]

        # now add replay buffer data
        for dir in trange(len(dirs), desc="Processing replay buffers"):
            experience = self._get_experience(
                dirs[dir], REPLAY_BUFFER_FILENAME, start, check_eval_metrics=False
            )
            if experience is None:
                continue
            obs_h5.append(experience["obs"])
            acts_h5.append(experience["acts"])
            poses_h5.append(jnp.array(experience["poses"]))
            start_idxs_h5.append(experience["start_idxs"])
            ep_lens.extend(experience["ep_lens"])
            start = experience["start"]

        dataset_metrics = {
            "num_episodes": len(ep_lens),
            "avg_episode_length": np.mean(ep_lens),
            "num_trajectories": len(start_idxs_h5),
        }
        dataset_metrics_path = os.path.join(
            self.replay_buffer_dir, Names.DATASET_METRICS_FILE_NAME
        )
        with open(dataset_metrics_path, "w") as f:
            json.dump(dataset_metrics, f, indent=4)

        obs_h5.close()
        acts_h5.close()
        poses_h5.close()
        start_idxs_h5.close()

        # just use the expert scaler params for the replay buffer data as well
        save_scaler_params(expert_scaler_params, self.replay_buffer_dir)

    def _save_data(
        self,
        all_data_listed: AllDataListed,
        output_dir: str,
        metrics: dict = {},
        poses_scaler_params: dict | None = None,
    ) -> dict:

        print("Saving actions...")
        all_acts = all_data_listed.pop("all_acts")
        acts_data = np.concatenate(all_acts, axis=0)
        acts_path = os.path.join(output_dir, Names.ACTIONS_FILE_NAME)
        acts_mean = np.mean(acts_data, axis=0)
        acts_std = np.std(acts_data, axis=0)
        with h5py.File(acts_path, "w") as acts_file:
            acts_file.create_dataset("data", data=acts_data)
        del acts_data, all_acts
        gc.collect()

        print("Saving poses...")
        all_poses = all_data_listed.pop("all_poses")
        poses_data = np.array(jnp.concatenate(all_poses, axis=0))
        poses_path = os.path.join(output_dir, Names.POSES_FILE_NAME)
        if poses_scaler_params is None:
            psp = math_utils.poses_scaler_params(
                all_poses,
                self.sys["poses"]["se2_idxs"],
                self.H,
                self.T,
            )
        else:
            psp = poses_scaler_params
        with h5py.File(poses_path, "w") as poses_file:
            poses_file.create_dataset("data", data=poses_data)
        del poses_data, all_poses
        gc.collect()

        print("Saving start indices...")
        all_start_idxs = all_data_listed.pop("start_idxs")
        start_idxs = np.concatenate(all_start_idxs, axis=0)
        num_trajs = len(start_idxs)
        print(f"Total number of trajectories: {num_trajs}")
        start_idxs_path = os.path.join(output_dir, Names.START_IDXS_FILE_NAME)
        with h5py.File(start_idxs_path, "w") as start_idxs_file:
            start_idxs_file.create_dataset("data", data=start_idxs)
        del start_idxs, all_start_idxs
        gc.collect()

        print("Saving observations...")
        obs_list: list[np.ndarray] = all_data_listed.pop("all_obs")
        obs_list.reverse()
        dataset_len = sum(len(ep) for ep in obs_list)
        obs_dim = obs_list[0].shape[-1]
        obs_path = os.path.join(output_dir, Names.OBS_FILE_NAME)
        with h5py.File(obs_path, "w") as obs_file:
            obs_file.create_dataset("data", shape=(dataset_len, obs_dim))
            idx = 0
            while obs_list:
                ep_obs = obs_list.pop()
                ep_len = ep_obs.shape[0]
                obs_file["data"][idx : idx + ep_len] = ep_obs
                del ep_obs
                gc.collect()
                idx += ep_len

            # TODO: we can compute these stats in the while loop w/o loading
            # the whole thing into RAM
            all_obs = obs_file["data"][:]
            obs_mean = np.mean(all_obs, axis=0)
            obs_std = np.std(all_obs, axis=0)
            del all_obs
            gc.collect()

        print("Saving scaler parameters...")
        scaler_params = {
            "obs": {
                "mean": obs_mean,
                "std": obs_std,
            },
            "acts": {
                "mean": acts_mean,
                "std": acts_std,
            },
            "poses": psp,
        }
        save_scaler_params(scaler_params, output_dir)

        metrics["num_trajectories"] = num_trajs
        dataset_metrics_path = os.path.join(output_dir, Names.DATASET_METRICS_FILE_NAME)
        with open(dataset_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        return scaler_params

    def _dirs_with_experience(self, data_dir: str, experience_filename: str):
        matching_dirs = []
        for root, dirs, files in os.walk(data_dir):
            if experience_filename in files:
                matching_dirs.append(root)
        return matching_dirs

    def _get_experience(
        self,
        dir: str,
        experience_filename: str,
        start: int,
        check_eval_metrics: bool = True,
    ) -> dict | None:
        if not self._is_good_episode(dir) and check_eval_metrics:
            return None

        experience_path = os.path.join(dir, experience_filename)
        try:
            with open(experience_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(
                f"WARNING: Could not load {experience_path} — {e.__class__.__name__}: {e}"
            )
            return None

        total_length = len(data)

        raw_obs = []
        acts = []
        raw_poses = []
        all_start_idxs = []

        obs_buffer = []
        acts_buffer = []
        poses_buffer = []
        ep_lens = []

        for i, t in enumerate(data):
            obs_buffer.append(t[OBS_IDX])
            acts_buffer.append(t[ACTS_IDX])
            poses_buffer.append(t[POSES_IDX])

            if t[TERMINATION_IDX] or i == total_length - 1:
                ep_obs = np.stack(obs_buffer)
                ep_acts = np.stack(acts_buffer)
                ep_poses = np.stack(poses_buffer)

                if self.cfg["pad_with_initial_state"]:
                    # Pad the beginning of each episode with the initial state
                    # for H steps
                    initial_obs = ep_obs[0]
                    initial_poses = ep_poses[0]
                    initial_acts = np.zeros_like(ep_acts[0])
                    ep_obs = np.concatenate(
                        (np.tile(initial_obs, (self.H, 1)), ep_obs), axis=0
                    )
                    ep_poses = np.concatenate(
                        (np.tile(initial_poses, (self.H, 1)), ep_poses), axis=0
                    )
                    ep_acts = np.concatenate(
                        (np.tile(initial_acts, (self.H, 1)), ep_acts), axis=0
                    )

                ep_len = len(ep_obs)
                ep_lens.append(ep_len)

                padding = self.H + self.T + self.end_padding + self.max_plan_interval
                ep_long_enough = ep_len >= padding

                any_bad_values = (
                    np.isnan(ep_obs).any()
                    or np.isinf(ep_obs).any()
                    or np.isnan(ep_acts).any()
                    or np.isinf(ep_acts).any()
                    or np.isnan(ep_poses).any()
                    or np.isinf(ep_poses).any()
                )

                if any_bad_values:
                    print(
                        f"WARNING: Episode in {dir} contains NaN or Inf values. "
                        f"Skipping this episode."
                    )

                if ep_long_enough and not any_bad_values:
                    # Store valid starting indices where the episode didn't
                    # terminate before the end of the trajectory
                    start_idxs = np.arange(
                        start,
                        start + ep_len - padding,
                    )
                    start += ep_len

                    raw_obs.append(ep_obs)
                    acts.append(ep_acts)
                    raw_poses.append(ep_poses)
                    all_start_idxs.append(start_idxs)

                obs_buffer = []
                acts_buffer = []
                poses_buffer = []

        # Concatenate episodes
        if len(raw_obs) == 0:
            return None
        raw_obs = np.concatenate(raw_obs, axis=0)
        acts = np.concatenate(acts, axis=0)
        raw_poses = np.concatenate(raw_poses, axis=0)
        all_start_idxs = np.concatenate(all_start_idxs, axis=0)

        # Convert raw observations to the system's observation format
        obs_components = []
        for obs_type in self.sys["obs"]["types"]:
            idx = obs_type["idx"]
            dim = obs_type["dim"]
            obs_slice = raw_obs[:, idx : idx + dim]
            obs_components.append(obs_slice)
        obs = np.concatenate(obs_components, axis=1)

        # Convert raw poses to the system's pose format
        poses_components = []
        for pose_type in self.sys["poses"]["types"]:
            idx = pose_type["idx"]
            dim = pose_type["dim"]
            poses_slice = raw_poses[:, idx : idx + dim]
            poses_components.append(poses_slice)
        poses = np.concatenate(poses_components, axis=1)

        return {
            "obs": obs,
            "acts": acts,
            "poses": poses,
            "start_idxs": all_start_idxs,
            "ep_lens": ep_lens,
            "start": start,
        }

    def _is_good_episode(self, dir: str) -> bool:
        with open(os.path.join(dir, RL_PARAMS_FILENAME), "r") as f:
            rl_params = yaml.safe_load(f)

        with open(os.path.join(dir, EVAL_METRICS_FILENAME), "r") as f:
            eval_metrics = json.load(f)

        env = rl_params["env"]

        if env == "step":
            crit = self.cfg["criteria"]["step"]
            return eval_metrics["eval/pos_x"] > crit["min_pos_x"]
        elif env == "ripples":
            crit = self.cfg["criteria"]["ripples"]
            return eval_metrics["eval/pos_x"] > crit["min_pos_x"]
        elif env == "rubble":
            crit = self.cfg["criteria"]["rubble"]
            return eval_metrics["eval/pos_x"] > crit["min_pos_x"]
        elif env == "slope":
            crit = self.cfg["criteria"]["slope"]
            return eval_metrics["eval/pos_x"] > crit["min_pos_x"]
        elif env == "flat":
            return True
        else:
            raise ValueError(f"Unknown environment type: {env}")


class _H5Appender:
    def __init__(
        self,
        path: str,
        key: str,
        feat_dim: int,
        dtype: np.dtype = np.float32,
    ):
        self.f = h5py.File(path, "w")
        self.feat_dim = feat_dim
        self.dtype = np.dtype(dtype)

        if self.feat_dim == 0:
            shape = (0,)
            maxshape = (None,)
        else:
            shape = (0, feat_dim)
            maxshape = (None, feat_dim)

        self.dset = self.f.create_dataset(
            key,
            shape=shape,
            maxshape=maxshape,
            dtype=self.dtype,
        )

    def __len__(self):
        return self.dset.shape[0]

    def append(self, batch: np.ndarray):
        if self.feat_dim == 0:
            n = self.dset.shape[0]
            assert batch.ndim == 1
        else:
            n, m = self.dset.shape
            assert batch.shape[1] == m
        self.dset.resize(n + batch.shape[0], axis=0)
        self.dset[n : n + batch.shape[0]] = batch.astype(self.dtype, copy=False)

    def close(self):
        self.f.flush()
        self.f.close()

from terra.data import Names, get_data_dir
from terra.utils.io import save_scaler_params
import terra.utils.config as config_utils
from terra.utils.data import generate_valid_indices
from terra.utils.math import poses_scaler_params
import numpy as np
import jax.numpy as jnp
import os
import h5py
import shutil


class DummyDataGenerator:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.num_samples = cfg["num_samples"]
        self.dataset_dir = get_data_dir(
            cfg["dataset_dir"],
            cfg["system"]["type"],
            cfg["history_length"],
            cfg["prediction_length"],
            False,
        )

        ob_dims = [ob["dim"] for ob in cfg["system"]["obs"]["types"]]
        self.obs_dim = config_utils.obs_dim_from_sys_config(cfg["system"])
        obs_noises = [ob["noise"] for ob in cfg["system"]["obs"]["types"]]
        self.obs_noise_stds = []
        for dim, noise in zip(ob_dims, obs_noises):
            self.obs_noise_stds.extend([noise] * dim)
        self.obs_noise_stds = np.array(self.obs_noise_stds)

        self.poses_dim = config_utils.pose_dim_from_sys_config(cfg["system"])
        self.acts_dim = cfg["system"]["acts"]["dim"]
        self.H = cfg["history_length"]
        self.T = cfg["prediction_length"]
        self.max_plan_interval = cfg["system"]["poses"]["max_plan_interval"]

    def generate(self):
        print("Generating dummy data...")
        print("Config:", self.cfg)
        os.makedirs(self.dataset_dir, exist_ok=True)

        print("Generating observations...")
        obs_data = np.random.normal(
            loc=0.0, scale=self.obs_noise_stds, size=(self.num_samples, self.obs_dim)
        )
        obs_mean = np.mean(obs_data, axis=0)
        obs_std = np.std(obs_data, axis=0)
        obs_path = os.path.join(self.dataset_dir, Names.OBS_FILE_NAME)
        with h5py.File(obs_path, "w") as obs_file:
            obs_file.create_dataset("data", data=obs_data)
        del obs_data

        print("Generating actions...")
        acts_data = np.random.normal(
            loc=0.0, scale=1.0, size=(self.num_samples, self.acts_dim)
        )
        acts_path = os.path.join(self.dataset_dir, Names.ACTIONS_FILE_NAME)
        acts_mean = np.mean(acts_data, axis=0)
        acts_std = np.std(acts_data, axis=0)
        with h5py.File(acts_path, "w") as acts_file:
            acts_file.create_dataset("data", data=acts_data)
        del acts_data

        print("Generating poses...")
        poses_data = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(self.num_samples, self.poses_dim),
        )
        poses_path = os.path.join(self.dataset_dir, Names.POSES_FILE_NAME)
        poses_mean = np.mean(poses_data, axis=0)
        poses_std = np.std(poses_data, axis=0)
        with h5py.File(poses_path, "w") as poses_file:
            poses_file.create_dataset("data", data=poses_data)

        print("Generating start indices...")
        start_indices = np.arange(
            0, self.num_samples - self.H - self.T - self.max_plan_interval
        )
        start_indices_path = os.path.join(self.dataset_dir, Names.START_IDXS_FILE_NAME)
        with h5py.File(start_indices_path, "w") as start_indices_file:
            start_indices_file.create_dataset("data", data=start_indices)

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
            "poses": {
                "mean": poses_mean,
                "std": poses_std,
            },
        }
        save_scaler_params(scaler_params, self.dataset_dir)

        # Dummy predictions for dynamics adaptation models
        dummy_pred_dir = os.path.join(self.dataset_dir, "predictions_dummy_model")
        os.makedirs(dummy_pred_dir, exist_ok=True)

        print("Generating observations predictions...")
        pred_obs_data = np.random.normal(
            loc=0.0,
            scale=self.obs_noise_stds,
            size=(self.num_samples, self.T, self.obs_dim),
        )
        pred_obs_path = os.path.join(dummy_pred_dir, Names.PREDICTED_OBS_FILE_NAME)
        with h5py.File(pred_obs_path, "w") as pred_obs_file:
            pred_obs_file.create_dataset("data", data=pred_obs_data)
        del pred_obs_data

        print("Generating poses predictions...")
        pred_poses_data = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(self.num_samples, self.T, self.poses_dim),
        )
        pred_poses_path = os.path.join(dummy_pred_dir, Names.PREDICTED_POSES_FILE_NAME)
        with h5py.File(pred_poses_path, "w") as pred_poses_file:
            pred_poses_file.create_dataset("data", data=pred_poses_data)
        del pred_poses_data

        print("Generating start indices for predictions...")
        start_indices_for_preds_path = os.path.join(
            dummy_pred_dir, Names.START_IDXS_FOR_PREDS_FILE_NAME
        )
        with h5py.File(start_indices_for_preds_path, "w") as start_indices_file:
            start_indices_file.create_dataset("data", data=start_indices)
        del start_indices

        expert_only_dir = os.path.join(
            os.path.dirname(self.dataset_dir), Names.EXPERT_DIR_NAME
        )
        os.makedirs(expert_only_dir, exist_ok=True)
        shutil.copytree(self.dataset_dir, expert_only_dir, dirs_exist_ok=True)

        print("Dummy data generation completed.")

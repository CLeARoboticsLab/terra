from terra.data import Names, get_data_dir
from terra.modeling import types
import terra.utils.config as config_utils
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
from collections import Counter
from typing import Any, TypedDict, Tuple, Optional


class DatasetItem(TypedDict):
    model_in: types.ModelInputs
    model_out: types.ModelOutputs
    metadata: dict[str, Any]


class DatasetBatch(TypedDict):
    model_in: types.ModelInputs
    model_out: types.ModelOutputs


class DatasetBase(Dataset):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._eval_mode = False
        self._dataset_dir = None

    def eval(self) -> None:
        self._eval_mode = True

    def train(self) -> None:
        self._eval_mode = False

    @property
    def dataset_dir(self) -> str:
        return self._dataset_dir

    @property
    def _can_add_noise(self) -> bool:
        return not self._eval_mode

    def __len__(self) -> int:
        raise NotImplementedError("This method must be implemented.")

    def __getitem__(self, idx: int) -> DatasetItem:
        raise NotImplementedError("This method must be implemented.")


class ErnestDatasetBase(DatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        # Load config
        self.hist_len = cfg["model"]["dataset"]["history_length"]
        self.T = cfg["model"]["dataset"]["prediction_length"]
        self.add_noise = cfg["data_mods"]["add_noise"]

        # Load data
        self._dataset_dir = cfg["data"]["dataset_dir"]
        if not os.path.isfile(os.path.join(self.dataset_dir, Names.OBS_FILE_NAME)):
            self._dataset_dir = get_data_dir(
                self.dataset_dir,
                cfg["system"]["type"],
                self.hist_len,
                self.T,
                cfg["model"]["dataset"]["expert_data_only"],
            )
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                f"Dataset directory '{self.dataset_dir}' does not exist."
            )
        start_idxs_path = os.path.join(self.dataset_dir, Names.START_IDXS_FILE_NAME)
        obs_path = os.path.join(self.dataset_dir, Names.OBS_FILE_NAME)
        acts_path = os.path.join(self.dataset_dir, Names.ACTIONS_FILE_NAME)
        self.start_idxs = h5py.File(start_idxs_path, "r")["data"]
        self.obs_data = h5py.File(obs_path, "r")["data"]
        self.acts_data = h5py.File(acts_path, "r")["data"]

        # Determine noise to add to observations
        ob_dims = [ob["dim"] for ob in cfg["system"]["obs"]["types"]]
        self.obs_dim = sum(ob_dims)
        obs_noises = [ob["noise"] for ob in cfg["system"]["obs"]["types"]]
        self.obs_noise_stds = []
        for dim, noise in zip(ob_dims, obs_noises):
            self.obs_noise_stds.extend([noise] * dim)
        self.obs_noise_stds = np.array(self.obs_noise_stds)

        np.random.seed(cfg["training"]["seed"])

        print(f"{len(self.start_idxs)/1e3:.1f}k trajectories loaded")

    @property
    def history_length(self) -> int:
        return self.hist_len

    @property
    def prediction_length(self) -> int:
        return self.T

    @property
    def num_points(self) -> int:
        """Total number of points (not trajectories) in the dataset"""
        return self.obs_data.shape[0]

    def __len__(self) -> int:
        return len(self.start_idxs)

    def __getitem__(self, idx: int) -> DatasetItem:
        raise NotImplementedError("This method must be implemented.")


class WorldModelDataset(ErnestDatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        # Load poses data
        poses_path = os.path.join(self.dataset_dir, Names.POSES_FILE_NAME)
        self.poses_data = h5py.File(poses_path, "r")["data"]

        # Determine noise to add to poses
        pose_dims = [pose["dim"] for pose in cfg["system"]["poses"]["types"]]
        poses_meas_noises = [
            pose["meas_noise"] for pose in cfg["system"]["poses"]["types"]
        ]
        self.poses_meas_noise_stds = []
        for dim, mean_noise in zip(pose_dims, poses_meas_noises):
            self.poses_meas_noise_stds.extend([mean_noise] * dim)
        self.poses_meas_noise_stds = np.array(self.poses_meas_noise_stds)

    def __getitem__(self, idx: int) -> DatasetItem:
        # metadata to be passed to subclasses
        t_H = self.start_idxs[idx]
        return self.get_item_by_t_H(t_H)

    def get_item_by_t_H(self, t_H: int) -> DatasetItem:
        # metadata to be passed to subclasses
        t = t_H + self.hist_len
        t_T = t + self.T
        metadata = {
            "t_H": t_H,
            "t": t,
            "t_T": t_T,
        }

        # inputs
        obs_hist = self.obs_data[t_H : t + 1]
        pose_hist = self.poses_data[t_H : t + 1]
        acts_hist = self.acts_data[t_H:t]
        fut_acts = self.acts_data[t:t_T]
        if self.add_noise and self._can_add_noise:
            obs_hist = _apply_noise(obs_hist, self.obs_noise_stds)
            pose_hist = _apply_noise(pose_hist, self.poses_meas_noise_stds)
        model_inputs: types.WorldModel.TrainingInputs = {
            "obs_hist": obs_hist,
            "pose_hist": pose_hist,
            "acts_hist": acts_hist,
            "fut_acts": fut_acts,
        }

        # outputs
        obs = self.obs_data[t + 1 : t_T + 1]
        poses = self.poses_data[t + 1 : t_T + 1]
        model_outputs: types.WorldModel.TrainingOutputs = {
            "obs": obs,
            "poses": poses,
            "pose_hist": pose_hist,
        }

        return {
            "model_in": model_inputs,
            "model_out": model_outputs,
            "metadata": metadata,
        }


class BehaviorCloningWithPoseHistDataset(ErnestDatasetBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)

        poses_path = os.path.join(self.dataset_dir, Names.POSES_FILE_NAME)
        self.poses_data = h5py.File(poses_path, "r")["data"]

        pose_dims = [pose["dim"] for pose in cfg["system"]["poses"]["types"]]
        poses_meas_noises = [
            pose["meas_noise"] for pose in cfg["system"]["poses"]["types"]
        ]
        self.poses_meas_noise_stds = []
        for dim, mean_noise in zip(pose_dims, poses_meas_noises):
            self.poses_meas_noise_stds.extend([mean_noise] * dim)
        self.poses_meas_noise_stds = np.array(self.poses_meas_noise_stds)

    def __getitem__(self, idx: int) -> DatasetItem:
        start_idx = self.start_idxs[idx]
        t_H = start_idx
        t = start_idx + self.hist_len
        t_T = t + self.T
        metadata = {
            "t_H": t_H,
            "t": t,
            "t_T": t_T,
        }

        obs_hist = self.obs_data[t_H : t + 1]
        acts_hist = self.acts_data[t_H:t]
        pose_hist = self.poses_data[t_H : t + 1]
        if self.add_noise and self._can_add_noise:
            obs_hist = _apply_noise(obs_hist, self.obs_noise_stds)
            pose_hist = _apply_noise(pose_hist, self.poses_meas_noise_stds)
        model_inputs: types.BehaviorCloningWithPoseHist.TrainingInputs = {
            "obs_hist": obs_hist,
            "pose_hist": pose_hist,
            "acts_hist": acts_hist,
        }

        actions = self.acts_data[t:t_T]
        model_outputs: types.BehaviorCloningWithPoseHist.TrainingOutputs = {
            "actions": actions,
        }

        return {
            "model_in": model_inputs,
            "model_out": model_outputs,
            "metadata": metadata,
        }


def _apply_noise(arr: np.ndarray, std: float | np.ndarray) -> np.ndarray:
    noise = np.random.randn(*arr.shape)
    return arr + noise * std


def collate_fn(batch: list[DatasetItem]) -> DatasetBatch:
    model_in_batch = {}
    model_out_batch = {}
    for key in batch[0]["model_in"].keys():
        model_in_batch[key] = np.stack([item["model_in"][key] for item in batch])
    for key in batch[0]["model_out"].keys():
        model_out_batch[key] = np.stack([item["model_out"][key] for item in batch])
    return {
        "model_in": model_in_batch,
        "model_out": model_out_batch,
    }


def collate_to_numpy_safe(
    batch: list[DatasetItem],
) -> Optional[Tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """
    Returns (model_in, model_out) where each is a dict[str, np.ndarray] stacked on axis 0.
    If no sample has consistent shapes across all keys, returns None (caller should skip).
    """

    def key_shapes(section: str) -> dict[str, list[Tuple[int, ...]]]:
        # For each key, collect shapes from all samples
        ks = batch[0][section].keys()
        return {k: [np.asarray(b[section][k]).shape for b in batch] for k in ks}

    def majority_shape(shapes: list[Tuple[int, ...]]) -> Tuple[int, ...]:
        # Mode of shapes; ties resolved arbitrarily
        return Counter(shapes).most_common(1)[0][0]

    # 1) Decide the target (majority) shape per key for both sections
    in_shapes = key_shapes("model_in")
    out_shapes = key_shapes("model_out")
    target_in = {k: majority_shape(v) for k, v in in_shapes.items()}
    target_out = {k: majority_shape(v) for k, v in out_shapes.items()}

    # 2) Keep only samples matching ALL keys' target shapes
    keep_idx = []
    bad_idx = []
    for i, sample in enumerate(batch):
        ok = True
        for k, tgt in target_in.items():
            if tuple(np.asarray(sample["model_in"][k]).shape) != tgt:
                ok = False
                break
        if ok:
            for k, tgt in target_out.items():
                if tuple(np.asarray(sample["model_out"][k]).shape) != tgt:
                    ok = False
                    break
        (keep_idx if ok else bad_idx).append(i)

    if bad_idx:
        print(
            f"[collate] dropped {len(bad_idx)}/{len(batch)} samples; "
            f"bad indices: {bad_idx[:8]}{'...' if len(bad_idx)>8 else ''}"
        )

    if not keep_idx:
        # Entire batch unusable—signal caller to skip
        return None

    # 3) Stack the kept samples; make sure outputs are owned & contiguous
    def stack_section(
        section: str, targets: dict[str, Tuple[int, ...]]
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for k, tgt_shape in targets.items():
            vals = [np.asarray(batch[i][section][k]) for i in keep_idx]
            # np.stack allocates a fresh buffer; enforce contiguous just in case
            arr = np.stack(vals, axis=0)
            if (not arr.flags["C_CONTIGUOUS"]) or (not arr.flags["WRITEABLE"]):
                arr = np.array(arr, copy=True)
            out[k] = arr
        return out

    model_in = stack_section("model_in", target_in)
    model_out = stack_section("model_out", target_out)
    return {
        "model_in": model_in,
        "model_out": model_out,
    }


def get_dataset(cfg: dict) -> DatasetBase:
    datasets = {
        "world_model": WorldModelDataset,
        "behavior_cloning_with_pose_hist": BehaviorCloningWithPoseHistDataset,
    }
    dataset_type = cfg["model"]["dataset"]["type"]
    if dataset_type not in datasets:
        raise ValueError(f"Dataset type '{dataset_type}' is not supported.")
    return datasets[dataset_type](cfg)

import os


class Names:
    EXPERT_DIR_NAME = "expert_only"
    ALL_EXPERIENCE_DIR_NAME = "all"
    OBS_FILE_NAME = "obs.hdf5"
    ACTIONS_FILE_NAME = "acts.hdf5"
    POSES_FILE_NAME = "poses.hdf5"
    START_IDXS_FILE_NAME = "start_indices.hdf5"
    SCALER_PARAMS_FILE_NAME = "scaler_params.json"
    DATASET_METRICS_FILE_NAME = "dataset_metrics.json"
    PREDICTED_OBS_FILE_NAME = "predicted_obs.hdf5"
    PREDICTED_POSES_FILE_NAME = "predicted_poses.hdf5"
    START_IDXS_FOR_PREDS_FILE_NAME = "start_indices_for_preds.hdf5"
    PREDICTED_DATASET_METRICS_FILE_NAME = "predicted_dataset_metrics.json"
    PROCESSED_DIR_FORMAT = "{}_H-{}_T-{}"


def get_data_dir(
    dataset_dir: str,
    system_name: str,
    history_len: int,
    prediction_len: int,
    expert_only: bool,
):
    """
    Returns the path to the processed dataset directory for a given system
    name, history length, and prediction length.
    """
    return os.path.join(
        dataset_dir,
        Names.PROCESSED_DIR_FORMAT.format(system_name, history_len, prediction_len),
        Names.EXPERT_DIR_NAME if expert_only else Names.ALL_EXPERIENCE_DIR_NAME,
    )


def get_data_dir_for_predictions(
    dataset_dir: str,
    system_name: str,
    history_len: int,
    prediction_len: int,
    model_name: str,
):
    """
    Returns the path to the processed dataset directory for predictions.
    """
    return os.path.join(
        get_data_dir(
            dataset_dir, system_name, history_len, prediction_len, expert_only=False
        ),
        f"predictions_{model_name}",
    )

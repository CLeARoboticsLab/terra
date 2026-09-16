from terra.data.data_generator import DummyDataGenerator
from terra.data.dataset import get_dataset
from terra.utils.io import load_scaler_params
import terra.utils.model as model_utils
from terra.modeling.models import get_model
import terra.modeling.models as models
import pytest
from hydra import initialize, compose
from omegaconf import OmegaConf
import flax.nnx as nnx
import numpy as np
import jax.numpy as jnp
import jax

REL_CONFIG_PATH = "../config"


@pytest.fixture
def config(request):
    model_name = request.param
    with initialize(config_path=REL_CONFIG_PATH, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"model={model_name}", f"loss={model_name}"],
        )
        cfg = OmegaConf.to_container(cfg, resolve=True)
        return cfg


@pytest.mark.parametrize(
    "config",
    [
        "world_model",
        "behavior_cloning_with_pose_hist",
    ],
    indirect=True,
)
def test_model(config: dict, tmp_path):
    # Separate Hydra context for dummy data generation
    with initialize(config_path=REL_CONFIG_PATH, version_base=None):
        gen_cfg = compose(
            config_name="generate_dummy_data",
            overrides=[f"system={config['system']['type']}"],
        )
        gen_cfg = OmegaConf.to_container(gen_cfg, resolve=True)

    # Configure dummy dataset
    gen_cfg["dataset_dir"] = str(tmp_path)
    gen_cfg["num_samples"] = 100
    gen_cfg["history_length"] = config["model"]["dataset"]["history_length"]
    gen_cfg["prediction_length"] = config["model"]["dataset"]["prediction_length"]

    # Generate the data
    data_generator = DummyDataGenerator(gen_cfg)
    data_generator.generate()

    # load the dataset and model
    config["data"]["dataset_dir"] = tmp_path
    dataset = get_dataset(config)
    scaler_params = load_scaler_params(dataset.dataset_dir)
    model = get_model(config, scaler_params, rngs=nnx.Rngs(config["training"]["seed"]))

    # test the model
    data = dataset[np.random.randint(0, len(dataset))]
    model_in, _ = model_utils.batch_to_model_in_out(data)
    model_in_batch = model_utils.repeat_along_batch_dim(model_in, 8)
    model(model_in_batch, deterministic=False)
    model_inf_fn = jax.jit(lambda x: model.inference(x))
    model_in_batch = model_utils.repeat_along_batch_dim(model_in, 1)
    model_inf_fn(model_in_batch)
    model_in_batch = model_utils.repeat_along_batch_dim(model_in, 2)
    model_inf_fn(model_in_batch)


def test_body_frame_poses():
    poses = jnp.array(
        [
            [-10, 10, np.deg2rad(135)],
            [-11, 11, np.deg2rad(145)],
            [-12, 12, np.deg2rad(155)],
            [-13, 13, np.deg2rad(180)],
            [-14, 14, np.deg2rad(180 + 45)],
            [-15, 15, np.deg2rad(180 + 90)],
            [-16, 16, np.deg2rad(180 + 135)],
        ]
    )
    se2_idxs = {"x": 0, "y": 1, "yaw": 2}

    x = {
        "pose_hist": poses[:4],
        "planner_poses": poses[4:],
    }
    x = model_utils.repeat_along_batch_dim(x, 1)
    x = models._body_frame_poses(x, se2_idxs)
    print(x)

    assert jnp.allclose(
        x["pose_hist"][0, :],
        jnp.array(
            [
                [0 * np.sqrt(2), 0.0, np.deg2rad(0)],
                [1 * np.sqrt(2), 0.0, np.deg2rad(10)],
                [2 * np.sqrt(2), 0.0, np.deg2rad(20)],
                [3 * np.sqrt(2), 0.0, np.deg2rad(45)],
            ]
        ),
        atol=1e-6,
    )
    assert jnp.allclose(
        x["planner_poses"][0, :],
        jnp.array(
            [
                [1, -1, np.deg2rad(45)],
                [2, -2, np.deg2rad(90)],
                [3, -3, np.deg2rad(135)],
            ]
        ),
        atol=1e-6,
    )


def test_loss_pred_highest_loss_bin_init(tmp_path):
    with initialize(config_path=REL_CONFIG_PATH, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["model=world_model", "loss=world_model"],
        )
        cfg = OmegaConf.to_container(cfg, resolve=True)
        gen_cfg = compose(
            config_name="generate_dummy_data",
            overrides=[f"system={cfg['system']['type']}"],
        )
        gen_cfg = OmegaConf.to_container(gen_cfg, resolve=True)

    gen_cfg["dataset_dir"] = str(tmp_path)
    gen_cfg["num_samples"] = 100
    gen_cfg["history_length"] = cfg["model"]["dataset"]["history_length"]
    gen_cfg["prediction_length"] = cfg["model"]["dataset"]["prediction_length"]

    cfg["loss_pred"] = {
        "thresholds": [0.5, 1.0],
        "init": {
            "type": "highest_loss_bin",
            "prior_prob": 0.99,
        },
        "attention": {
            "layers": 1,
            "heads": 4,
            "mlp_hidden_size_factor": 2,
            "mask": "causal",
        },
        "decoder": {
            "layers": 0,
            "mlp_hidden_size_factor": 1,
        },
    }

    data_generator = DummyDataGenerator(gen_cfg)
    data_generator.generate()

    cfg["data"]["dataset_dir"] = tmp_path
    dataset = get_dataset(cfg)
    scaler_params = load_scaler_params(dataset.dataset_dir)
    world_model = models.WorldModel(cfg, scaler_params, rngs=nnx.Rngs(0))
    loss_pred_model = models.WorldModelWithLossPredTransformer(
        world_model, cfg, rngs=nnx.Rngs(1)
    )

    data = dataset[np.random.randint(0, len(dataset))]
    model_in, _ = model_utils.batch_to_model_in_out(data)
    model_in_batch = model_utils.repeat_along_batch_dim(model_in, 8)
    y = loss_pred_model(model_in_batch, deterministic=True)
    probs = jax.nn.softmax(y["loss_pred_logits"], axis=-1)

    assert jnp.all(jnp.argmax(probs, axis=-1) == loss_pred_model.num_bins - 1)
    assert jnp.allclose(probs[..., -1], 0.99, atol=1e-6)
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-6)



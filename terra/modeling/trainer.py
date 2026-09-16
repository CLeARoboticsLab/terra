from terra.data.dataset import get_dataset, DatasetBase, collate_to_numpy_safe
from terra.modeling.models import get_model
from terra.utils.io import load_scaler_params
from terra.utils.model import batch_to_model_in_out, load_model_from_ckpt, count_params
from terra.modeling.losses import get_loss
from terra.modeling import types
from terra.modeling import CONFIG_FILE_NAME
from torch.utils.data import random_split, get_worker_info, DataLoader
import orbax.checkpoint as ocp
import optax
import datetime
import wandb
import torch
import os
import yaml
import flax.nnx as nnx
import time
from tqdm import tqdm


def train(cfg: dict):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{date_time}" if cfg["run_name"] is None else cfg["run_name"]
    train_steps = 0

    max_steps = None
    if "max_steps" in cfg["training"] and cfg["training"]["max_steps"] > 0:
        max_steps = cfg["training"]["max_steps"]
        print(f"Training will stop after {max_steps} steps.")

    # start wandb
    if cfg["wandb"]["log"]:
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"]["entity"],
            name=run_name,
            config=cfg,
        )
    print("Running training with config: ", cfg)

    # Create datasets
    dataset = get_dataset(cfg)
    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_dataset, test_dataset = random_split(
        dataset,
        [
            cfg["training"]["training_data_ratio"],
            1 - cfg["training"]["training_data_ratio"],
        ],
        generator=generator,
    )

    # holds the current training mode
    training = True

    # used to update each worker's dataset each time all of the data is seen
    def worker_init_fn(worker_id):
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset: DatasetBase = worker_info.dataset.dataset
            if training:
                dataset.train()
            else:
                dataset.eval()

    # create dataloaders
    train_set = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        shuffle=True,
        num_workers=cfg["data"]["num_train_workers"],
        worker_init_fn=worker_init_fn,
        collate_fn=collate_to_numpy_safe,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )
    test_set = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        drop_last=True,
        num_workers=cfg["data"]["num_test_workers"],
        worker_init_fn=worker_init_fn,
        collate_fn=collate_to_numpy_safe,
        prefetch_factor=cfg["data"]["prefetch_factor"],
        persistent_workers=False,
        generator=generator,
    )

    # load scaler params
    scaler_params = load_scaler_params(dataset.dataset_dir)

    ckpt_dir = None
    if (
        "resume_checkpoint" in cfg["checkpoint"]
        and cfg["checkpoint"]["resume_checkpoint"] is not None
    ):
        # load model from checkpoint if specified
        print("Resuming from checkpoint...")
        ckpt_dir = cfg["checkpoint"]["resume_checkpoint"]
        model, model_cfg, train_steps = load_model_from_ckpt(ckpt_dir)
        cfg["model"] = model_cfg["model"]  # use the model cfg from the checkpoint
        if max_steps is not None:
            max_steps += train_steps
    else:
        # otherwise, create a new model
        print("Creating a new model...")
        model = get_model(cfg, scaler_params, rngs=nnx.Rngs(cfg["training"]["seed"]))

    # Setup checkpointing with Orbax
    if cfg["checkpoint"]["enabled"]:
        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg["checkpoint"]["max_to_keep"],
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=False,
        )
        if ckpt_dir is None:
            ckpt_dir = os.path.join(cfg["checkpoint"]["checkpoint_dir"], run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(os.path.join(ckpt_dir, CONFIG_FILE_NAME), "w") as f:
                struct = {}
                for k, v in scaler_params.items():
                    struct[k] = len(v["mean"])
                cfg["scaler_params_struct"] = struct
                yaml.dump(cfg, f, default_flow_style=False)

    # Get the loss function;
    loss_fn = get_loss(cfg)

    # Freeze LoRA parameters if they exist (they will be trained online)
    filt = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    diff_state = nnx.DiffState(0, filt)
    print(f"Parameters to be optimized: {count_params(model, filt)}")
    print(f"Total parameters: {count_params(model, nnx.Param)}")

    # set up optimizer and metrics
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        decay_steps=cfg["training"]["decay_steps"],
        end_value=cfg["training"]["end_learning_rate"],
    )

    optimizer = nnx.Optimizer(model, optax.adam(lr_sched), wrt=filt)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        **{k: nnx.metrics.Average(k) for k in loss_fn.loss_terms},
    )
    metrics.reset()

    @nnx.jit
    def train_step(
        model,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        x: types.ModelInputs,
        y: types.ModelOutputs,
    ):
        grad_fn = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)
        (loss, losses), grads = grad_fn(model, x, y, False)
        metrics.update(loss=loss, **losses)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def eval_step(
        model,
        metrics: nnx.MultiMetric,
        x: types.ModelInputs,
        y: types.ModelOutputs,
    ):
        loss, losses = loss_fn(model, x, y, True)
        metrics.update(loss=loss, **losses)
        return loss

    print("Starting training")
    num_epochs = cfg["training"]["num_epochs"]
    metrics_dict = {}
    interval_start_time = time.time()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_start_time = time.time()

        # train
        training = True
        pbar_train = tqdm(train_set, desc="Training", unit="batch")
        for i, batch in enumerate(pbar_train):
            if batch is None:
                continue
            x, y = batch_to_model_in_out(batch)
            loss = train_step(model, optimizer, metrics, x, y)
            pbar_train.set_description(f"Training (Loss: {float(loss):.4f})")
            train_steps += 1

            is_last_step = i == len(train_set) - 1 and epoch == num_epochs - 1
            steps_done = max_steps is not None and train_steps >= max_steps
            if (
                train_steps % cfg["training"]["eval_interval"] == 0
                or is_last_step
                or steps_done
            ):
                interval_time = time.time() - interval_start_time
                metrics_dict["steps_per_second"] = (
                    cfg["training"]["eval_interval"] / interval_time
                )

                for metric, value in metrics.compute().items():
                    metrics_dict[f"train/{metric}"] = float(value)
                metrics.reset()

                # test
                training = True
                pbar_test = tqdm(test_set, desc="Testing", unit="batch")
                for batch in pbar_test:
                    if batch is None:
                        continue
                    x, y = batch_to_model_in_out(batch)
                    loss = eval_step(model, metrics, x, y)
                    pbar_test.set_description(f"Testing (Loss: {float(loss):.4f})")

                for metric, value in metrics.compute().items():
                    metrics_dict[f"test/{metric}"] = float(value)
                metrics.reset()

                if cfg["checkpoint"]["enabled"]:
                    state = nnx.state(model)
                    pure_dict_state = state.to_pure_dict()
                    with ocp.CheckpointManager(ckpt_dir, options=ckpt_options) as mngr:
                        mngr.save(
                            train_steps,
                            args=ocp.args.StandardSave(pure_dict_state),
                            metrics=metrics_dict,
                        )

                metrics_dict["epoch"] = epoch

                # Wandb logging
                if cfg["wandb"]["log"]:
                    wandb.log(metrics_dict, step=train_steps)

                print(f"Steps: {train_steps}, Metrics: {metrics_dict}")
                metrics_dict = {}
                interval_start_time = time.time()

            if steps_done:
                print(f"Stopping training after {train_steps} steps.")
                break

        if steps_done:
            break

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.2f} seconds.")

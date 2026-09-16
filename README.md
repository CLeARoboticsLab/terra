# TERRA -- Targeted Exploration for Reward-Relevant Adaptation

## Setup

Create a new conda environment:

```bash
conda create -n terra python=3.10
conda activate terra
```

Clone and install the repository:

```bash
git clone https://github.com/CLeARoboticsLab/terra.git
cd terra
pip install -e .[training]
```

Install the appropriate version of JAX for your system; see [JAX installation instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation). For example, for NVIDIA GPUs:

```bash
pip install -U "jax[cuda12]"
```

## Data Processing

### From RL experience buffers (full pipeline)

Before training, the dataset must be built from the RL experience buffers with the below command. Additional configuration is found in `config/process_data.yaml`. Specify the following:

- `data_dir`: The directory containing the RL experience buffers. This directory will be recursively scanned for all experience pkl files.
- `output_dir`: The directory where the processed dataset will be saved.

```bash
python scripts/process_data.py \
    data_dir=<path_to_rl_experience_buffers> \
    output_dir=<path_to_save_processed_dataset>
```

### Dummy data (no external RL repo)

Since the RL pipeline is not included in this repository, you can generate dummy data for testing the training pipeline with the below command. Additional configuration is found in `config/generate_dummy_data.yaml`. Specify the following:

- `dataset_dir`: The directory where the dummy dataset will be saved.
- `num_samples`: The number of samples to generate.

```bash
python scripts/generate_dummy_data.py \
    dataset_dir=<path_to_save_dummy_dataset> \
    num_samples=1000000
```

## Training

Perform training with the below command. Additional configuration is found in `config/train.yaml` file. Specify the following:

- `run_name`: An optional name for the training run. If not specified, a default name will be generated.
- `data.dataset_dir`: The path to the dataset directory.
- `checkpoint.checkpoint_dir`: The directory to save checkpoints to.

```bash
python scripts/train.py run_name=<optional_name_for_run> \
    run_name=<optional_name_for_run> \
    data.dataset_dir=<path_to_processed_dataset> \
    checkpoint.checkpoint_dir=<path_to_save_checkpoints>
```

## Deployment

For a runnable, documented walkthrough of loading BC and world-model checkpoints,
running MPPI with online adaptation and **loss prediction**, see:

- [`examples/deploy_mppi_adaptation_loss_pred/README.md`](examples/deploy_mppi_adaptation_loss_pred/README.md)

Quick run (uses **dummy** checkpoints committed in the repo — not for real robots):

```bash
python examples/deploy_mppi_adaptation_loss_pred/deploy_mppi_adaptation_loss_pred.py
```

Train your own checkpoints with the [Training](#training) section, then pass
`--bc-checkpoint` and `--world-model-checkpoint` to the example script.

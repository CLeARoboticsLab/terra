import hydra
from omegaconf import DictConfig, OmegaConf
from terra import CONFIG_DIR
from terra.data.dataset import get_dataset, collate_fn
from tqdm import tqdm


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    dataset = get_dataset(cfg)
    first_item = dataset[0]
    first_batch = collate_fn([first_item])

    def _print_shapes(prefix, obj):
        if hasattr(obj, "shape"):
            print(f"{prefix}: {tuple(obj.shape)}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _print_shapes(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _print_shapes(f"{prefix}[{i}]", v)

    print("First batch shapes:")
    _print_shapes("", first_batch)

    for i in tqdm(range(len(dataset))):
        batch = collate_fn([dataset[i]])
        check_batch(first_batch, batch, i)


def check_batch(first_batch, batch, idx):
    check_item(first_batch["model_in"], batch["model_in"], idx)
    check_item(first_batch["model_out"], batch["model_out"], idx)


def check_item(first_item, item, idx):
    for k, v in first_item.items():
        if v.shape != item[k].shape:
            print(
                f"Shape mismatch at index {idx} for key {k}: {v.shape} vs {item[k].shape}"
            )


if __name__ == "__main__":
    main()

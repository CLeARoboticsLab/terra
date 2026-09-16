import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"

import hydra
from omegaconf import DictConfig, OmegaConf
from terra.modeling import trainer
from terra import CONFIG_DIR


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    trainer.train(cfg)


if __name__ == "__main__":
    main()

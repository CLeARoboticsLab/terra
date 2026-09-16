from terra.data.data_generator import DummyDataGenerator
from terra import CONFIG_DIR
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(
    config_path=CONFIG_DIR, config_name="generate_dummy_data", version_base=None
)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    data_generator = DummyDataGenerator(cfg)
    data_generator.generate()


if __name__ == "__main__":
    main()

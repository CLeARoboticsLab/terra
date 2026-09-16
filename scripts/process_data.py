from terra.data.data_processor import DataProcessor
from terra import CONFIG_DIR
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(
    config_path=CONFIG_DIR, config_name="process_data", version_base=None
)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    data_processor = DataProcessor(cfg)
    data_processor.process()


if __name__ == "__main__":
    main()

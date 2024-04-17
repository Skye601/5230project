from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import webdataset as wds
from hmr2.datasets.image_dataset import ImageDataset
import hydra
from omegaconf import DictConfig, OmegaConf


# MPI-INF-TRAIN-PRUNED:
#     TYPE: ImageDataset
#     URLS: hmr2_training_data/dataset_tars/mpi-inf-train-pruned/{000000..00006}.tar
#     epoch_size: 12_000

@hydra.main(version_base="1.2", config_path=str(root/"hmr2/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    dataset = ImageDataset.load_tars_as_webdataset(cfg, url, train=True)


if __name__ == "__main__":

    url= 'hmr2_training_data/dataset_tars/mpi-inf-train-pruned/{000000..00006}.tar'
    main()
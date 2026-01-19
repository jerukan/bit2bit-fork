from spadio import SPADFolder, SPADData  # noqa
from spadclean import GenerateTestData, SPADHotpixelTool  # noqa
from pathlib import Path
from utils import clean_hotpixels
from inference import cpu_inference
from metadata import TrainData, ModelConfig, TrainConfig, load_config
from dataset import BernoulliDataset3D, BinomDataset3D, N2NDataset3D  # noqa
from spadgapmodels import SPADGAP
import torch
import torch.utils.data as dt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from tifffile import imwrite, imread
import numpy as np
import logging
import sys


config = load_config(path=Path("./config.yml"))  # CLI argument

logging.basicConfig(
    filename=config["PATH"]["logger"],
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

data_type = config["PATH"]["data_type"]
if data_type not in ["raw", "processed"]:
    raise ValueError("Data type must be RAW or CLEAN")

dir_path = Path(config["PATH"]["dir_path"])
num_of_files = config["PATH"]["num_of_files"]

data_dir = Path(config["PATH"]["data_dir"])
data_path = config["PATH"]["data_path"]
data_file = config["PATH"]["data_file"]
ground_truth_path = config["PATH"]["ground_truth_path"]
ground_truth_file = config["PATH"]["ground_truth_file"]

data_path = data_dir / data_file if data_path == "" else Path(data_path)
ground_truth_path = (
    data_dir / ground_truth_file if ground_truth_path == "" else Path(ground_truth_path)
)

if data_type == "raw":
    try:
        input_folder = SPADFolder(dir_path)
    except FileNotFoundError:
        logging.error("Folder not found")
        sys.exit(1)
    input = input_folder.spadstack[:num_of_files]
    data = input.process(clean_hotpixels)
    del input
elif data_type == "processed":
    try:
        data = imread(data_path)
        ground_truth_file = imread(ground_truth_path)
    except FileNotFoundError:
        logging.error("File not found")
        sys.exit(1)


data_config = TrainData.from_config(config["DATA"], data.astype(np.float32))
model_config = ModelConfig.from_config(config["MODEL"])
train_config = TrainConfig.from_config(config["TRAINING"], data_config, model_config)
train_config.metadata()
val_data_config = TrainData.from_config(config["DATA"], data.astype(np.float32))
val_data_config.random_crop = False
val_data_config.metadata()

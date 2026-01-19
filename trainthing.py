# import torch, traceback
# orig = getattr(torch._C, "_cuda_init", None)
# def _wrapped_cuda_init():
#     print("=== torch._C._cuda_init called ===")
#     traceback.print_stack(limit=8)
#     if orig:
#         orig()
# torch._C._cuda_init = _wrapped_cuda_init


from tqdm.auto import tqdm
from spadio import SPADFolder, SPADData  # noqa
from spadclean import GenerateTestData, SPADHotpixelTool  # noqa
from pathlib import Path
from utils import clean_hotpixels
from inference import cpu_inference
from metadata import TrainData, ModelConfig, TrainConfig, load_config
from dataset import (
    BernoulliDataset3D,
    ValidationDataset3D,
    PairedDataset,
    BinomDataset3D,
    N2NDataset3D,
)  # noqa
from spadgapmodels import SPADGAP
import torch
import torch.utils.data as dt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from tifffile import imwrite, imread
import numpy as np
import logging
import sys
import shutil

def thin_frames_uniform(frames, keep_prob, seed=None):
    rng = np.random.default_rng(seed=seed)
    frames = frames.copy()
    nframes = frames.shape[0]
    # chunk over time to avoid constructing huge intermediate arrays
    chunk_size = 1000
    for t0 in tqdm(list(range(0, nframes, chunk_size)), desc="Thinning full frame chunks over time"):
        t1 = min(t0 + chunk_size, nframes)
        sub = frames[t0:t1]  # view into wholetotalbit
        eventswhere = np.nonzero(sub)
        nevents = eventswhere[0].shape[0]
        if nevents == 0:
            continue
        # mask of events to delete
        mask = rng.random(nevents) > keep_prob
        sub[eventswhere[0][mask], eventswhere[1][mask], eventswhere[2][mask]] = 0
    return frames

configure_path = Path("./config.yml")
config = load_config(path=configure_path)  # CLI argument

# logging.basicConfig(
#     # filename=config["PATH"]["logger"],
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     stream=sys.stdout,
# )

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
model_path = Path(config["PATH"]["model_path"])

data_path = data_dir / data_file if data_path == "" else Path(data_path)
ground_truth_path = (
    data_dir / ground_truth_file if ground_truth_path == "" else Path(ground_truth_path)
)
if data_type == "raw":
    try:
        if dir_path.is_dir():
            input_folder = SPADFolder(dir_path)
            input = input_folder.spadstack[:num_of_files]
            data = input.process(clean_hotpixels)
        else:
            input_folder = SPADData(dir_path)
            input = input_folder.data
            data = clean_hotpixels(input)
    except FileNotFoundError:
        logging.error("Folder not found")
        # sys.exit(1)
    del input
elif data_type == "processed":
    try:
        data = imread(data_path)
        ground_truth_file = imread(ground_truth_path)
    except FileNotFoundError:
        logging.error("File not found")
        # sys.exit(1)

data = data[:40000]
keep_prob = config["PATH"]["thin"]
if keep_prob < 1.0:
    data = thin_frames_uniform(data, keep_prob=keep_prob, seed=42)
idx_train = int(data.shape[0] * 0.8)
traindata = data[:idx_train]
valdata = data[idx_train:]
data_config = TrainData.from_config(config["DATA"], traindata.astype("float32"))
model_config = ModelConfig.from_config(config["MODEL"])
train_config = TrainConfig.from_config(config["TRAINING"], data_config, model_config)
val_data_config = TrainData.from_config_validation(
    config["DATA"], (valdata.astype(np.float32))
)
print(train_config.metadata())

train_data = BernoulliDataset3D.from_dataclass(data_config)
val_data = ValidationDataset3D.from_dataclass(val_data_config)

loader_config = {
    "batch_size": train_config.batch_size,
    "shuffle": train_config.shuffle,
    "pin_memory": train_config.pin_memory,
    "drop_last": train_config.drop_last,
    "num_workers": train_config.num_workers,
    "persistent_workers": True,
}

train_loader = dt.DataLoader(train_data, **loader_config)
loader_config["shuffle"] = False
val_loader = dt.DataLoader(val_data, **loader_config)

test_name = train_config.name
default_root_dir = model_path / test_name
if not default_root_dir.exists():
    default_root_dir.mkdir(parents=True)

model = SPADGAP.from_dataclass(model_config)
model.train()

logger = TensorBoardLogger(save_dir=model_path, name=test_name)

trainer = pl.Trainer(
    default_root_dir=default_root_dir,
    accelerator="gpu",
    gradient_clip_val=1,
    precision=train_config.precision,  # type: ignore
    devices=[0, 1, 2, 3, 4, 5, 6, 7],
    strategy="ddp_find_unused_parameters_true",
    max_epochs=train_config.epochs,
    callbacks=[
        ModelCheckpoint(
            save_weights_only=True,
            mode="min",
            monitor="val_loss",
            save_top_k=2,
        ),
        LearningRateMonitor("epoch"),
        # EarlyStopping("val_loss", patience=25),
        # DeviceStatsMonitor(),
    ],
    logger=logger,  # type: ignore
    profiler="simple",
    limit_val_batches=20,
    enable_model_summary=True,
    enable_checkpointing=True,
)
# print(f"input_size: {tuple(next(iter(train_loader))[0].shape)}")
print(f"file: {test_name}")

model.train()
train_config.to_yaml(default_root_dir / "metadata.yml")
shutil.copyfile(configure_path, default_root_dir / "config.yml")
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint(default_root_dir / "final_model.ckpt")

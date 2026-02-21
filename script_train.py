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
import dask.array as da
import zarr

def read_quanta_zarr(path, load_data=True):
    """
    Reads quanta data from a Zarr file and returns points and quantaframes.

    Args:
        path (str or Path): Path to the Zarr file.
        load_data (bool): if True, loads all arrays into memory.

    Returns:
        tuple:
            - quantaframes: array of shape (T, H, W) with binary frames. If
                load_data is False, the array will be lazy-loaded.
            - (t, y, x) (np.ndarray): tuple of 1D arrays for t, y, and x coordinates.
            - attrs (dict): relevant metadata (fps and T_exp).
    """
    zarrdata = zarr.open_group(path, mode="r")
    framekey = "frames" if "frames" in zarrdata else "quantaframes"
    contains_coords = all(k in zarrdata for k in ("t", "y", "x"))
    quantaframes = zarrdata[framekey]
    if contains_coords:
        t = zarrdata["t"]
        y = zarrdata["y"]
        x = zarrdata["x"]
    else:
        t = y = x = None
    if load_data:
        quantaframes = quantaframes[:]
        if contains_coords:
            t = t[:]
            y = y[:]
            x = x[:]
    return quantaframes, (t, y, x), zarrdata.attrs

def prob_from_dcr(dcr_rate_hz, fps):
    """
    Convert a dark count rate in Hz to a per-frame probability of a dark count photon.
    """
    return 1 - np.exp(-dcr_rate_hz / fps)

def thin_frames_uniform(frames, keep_prob, dcr_prob=None, seed=None):
    """
    Thin binary frames uniformly with probability keep_prob. Also adds dark
    count photons to lower the SNR.

    This is an expensive operation on SPAD data, so dask is used for
    multiprocessing.

    Args:
        dcr_prob: dark count photon probability (not the rate itself)
    """
    T, H, W = frames.shape
    # convert to a dask array with automatic chunking and apply a lazy random mask
    frames = da.from_array(frames, chunks=(400, H, W))
    rs = da.random.RandomState(seed)
    mask = rs.random_sample(frames.shape, chunks=frames.chunks) < keep_prob
    if dcr_prob is not None:
        dcr_photons = rs.binomial(1, dcr_prob, size=frames.shape, chunks=frames.chunks).astype("uint8")
    else:
        dcr_photons = da.zeros(frames.shape, chunks=frames.chunks, dtype="uint8")
    frames = (frames.astype("uint8") & mask.astype("uint8")) | dcr_photons
    return frames.compute()

configure_path = Path("./config.yml")
config = load_config(path=configure_path)  # CLI argument

datanames = [
    "teaser-gunballoon-dark-acq00002"
]
slices_interest = [
    slice(60000, 100000)
]
keep_probs = [1.0, 1/10]

# logging.basicConfig(
#     # filename=config["PATH"]["logger"],
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     stream=sys.stdout,
# )

for i, dataname in enumerate(datanames):
    data_type = config["PATH"]["data_type"]
    if data_type not in ["raw", "processed", "zarr"]:
        raise ValueError("Data type must be RAW or CLEAN or ZARR")

    # data_path = Path(config["PATH"]["data_path"])
    data_path = Path("/scratch/jeyan/photon-testing-grounds/data/quanta") / f"{dataname}.zarr"
    num_of_files = config["PATH"]["num_of_files"]
    data_file = config["PATH"]["data_file"]
    model_path = Path(config["PATH"]["model_path"])

    if data_type == "raw":
        try:
            if data_path.is_dir():
                input_folder = SPADFolder(data_path)
                input = input_folder.spadstack[:num_of_files]
                data_orig = input.process(clean_hotpixels)
            else:
                input_folder = SPADData(data_path)
                input = input_folder.data
                data_orig = clean_hotpixels(input)
        except FileNotFoundError:
            logging.error("Folder not found")
            # sys.exit(1)
        del input
    elif data_type == "zarr":
        data_orig, _, _ = read_quanta_zarr(data_path)

    slice_interest = slices_interest[i]
    FRAME_LIMIT = 40000
    if slice_interest is not None:
        data_orig = data_orig[slice_interest]
    else:
        data_orig = data_orig[:FRAME_LIMIT]
    for keep_prob in keep_probs:
        # keep_prob = config["PATH"]["thin"]
        if keep_prob < 1.0:
            dcr_prob = prob_from_dcr(dcr_rate_hz=25, fps=100000)
            data = thin_frames_uniform(data_orig, keep_prob=keep_prob, dcr_prob=dcr_prob, seed=42)
        else:
            data = data_orig
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

        # test_name = train_config.name
        test_name = f"{data_path.stem}-thin{keep_prob:.3f}"
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
            devices=[1, 2, 3, 4, 5, 6, 7],
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

        del trainer
        del model

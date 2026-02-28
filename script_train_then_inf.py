# import torch, traceback
# orig = getattr(torch._C, "_cuda_init", None)
# def _wrapped_cuda_init():
#     print("=== torch._C._cuda_init called ===")
#     traceback.print_stack(limit=8)
#     if orig:
#         orig()
# torch._C._cuda_init = _wrapped_cuda_init

import datetime
import tempfile
import os
import gc
import time
from tqdm.auto import tqdm
from spadio import SPADFolder, SPADData  # noqa
from spadclean import GenerateTestData, SPADHotpixelTool  # noqa
from pathlib import Path
from utils import clean_hotpixels
import matplotlib.pyplot as plt
import cv2
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
import torch.distributed as dist
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
from inference import gpu_patch_inference
import zarr

def get_codec_for_format(format: str):
    """
    Get appropriate fourcc codec string for given video format.
    """
    format = format.lower()
    if format == "mp4":
        return "mp4v"
    elif format == "avi":
        return "FFV1"
    elif format == "mov":
        return "avc1"
    else:
        raise ValueError(f"I haven't added the codec for: {format}")

def to_video(
    frames: np.ndarray, path, res_scale=1.0, playback_fps=None, gamma=1.0, cmap=None, fileformat=None,
    vmin=None, vmax=None, quantile=None, framenames=None
):
    """
    Saves video frame arrays to a video file or sequence of PNGs. If path has no extension, 
    it is treated as a directory and individual image files are saved.

    Args:
        frames (np.ndarray): (T x H x W x C) (RGB) or (T x H x W) (intensity) video frames.
        path (str or Path): output video file path or directory for image files.
        res_scale (float): resolution scaling factor with nearest neighbor interpolation.
        cmap: ignored if frames are RGB; otherwise, matplotlib colormap name or object.
        fileformat (str or None): video format (e.g., "mp4", "avi"), or image format (e.g., "png");
            if None, inferred from path suffix.
        quantile (float or None): if not None, use quantiles to determine vmin and vmax for normalization
            (ignored if vmin or vmax are specified).
    """
    path = Path(path)
    if cmap is None:
        cmap = "viridis"
    cmap_fn = plt.get_cmap(cmap)
    is_rgb = False
    if frames.ndim == 4:
        if frames.shape[3] == 3:
            is_rgb = True
        else:
            raise ValueError("4D frames array must have shape (T, H, W, 3) for RGB video")
    elif frames.ndim == 3:
        is_rgb = False
    else:
        raise ValueError("frames must be a 3D or 4D numpy array")

    # compute a normalized intensity in [0,1] for colormap input
    if vmax is None:
        if quantile is not None:
            vmax = float(np.quantile(frames, quantile))
        else:
            vmax = float(np.max(frames))
    if vmin is None:
        if quantile is not None:
            vmin = float(np.quantile(frames, 1 - quantile))
        else:
            vmin = float(np.min(frames))
            if vmin >= 0:
                print(f"vmin was not specified and frames have non-negative values, so using vmin=0 for more accurate scaling")
                vmin = 0.0

    H, W = frames.shape[1], frames.shape[2]
    if res_scale != 1.0:
        out_W = int(W * res_scale)
        out_H = int(H * res_scale)
    else:
        out_W = W
        out_H = H
    # if path is a directory, write individual image files
    is_video_file = path.suffix in [".mp4", ".avi", ".mov", ".mkv"]
    if not is_video_file:
        path.mkdir(parents=True, exist_ok=True)
        if fileformat is None:
            fileformat = "png"
    else:
        if playback_fps is None:
            raise ValueError("playback_fps must be specified if saving a video file")
        path.parent.mkdir(parents=True, exist_ok=True)
        if fileformat is None:
            fileformat = path.suffix[1:].lower()
        codec = get_codec_for_format(fileformat)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vidwriter = cv2.VideoWriter(str(path), fourcc, playback_fps, (out_W, out_H), isColor=True)

    max_frames = len(frames)

    if not is_video_file:
        allpaths = []
    for i in tqdm(range(max_frames), desc="Writing video frames"):
        intensity = (np.clip(frames[i], vmin, vmax) - vmin) / (vmax - vmin)  # normalize to [0,1]
        if gamma != 1:
            intensity = intensity ** gamma
        if is_rgb:
            rgb_mapped = (intensity * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        else:
            # apply matplotlib colormap -> returns RGBA in [0,1]
            rgba_mapped = cmap_fn(intensity)  # shape (H,W,4)
            rgb_mapped = (rgba_mapped[..., :3] * 255.0).astype(np.uint8)  # (H,W,3) in RGB
        bgr_mapped = rgb_mapped[..., ::-1]  # convert to BGR for OpenCV
        if res_scale != 1.0:
            bgr_mapped = cv2.resize(bgr_mapped, (out_W, out_H), interpolation=cv2.INTER_NEAREST)
        if is_video_file:
            vidwriter.write(bgr_mapped)
        else:
            if framenames is None:
                frame_path = path / f"frame_{i:05d}.{fileformat}"
            else:
                frame_path = path / f"{framenames[i]}.{fileformat}"
            cv2.imwrite(str(frame_path), bgr_mapped)
            allpaths.append(frame_path)
    if is_video_file:
        vidwriter.release()
    if not is_video_file:
        return allpaths
    return path

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


def zip_run(results_dir: str, out_dir: str, run_id: str | None = None) -> Path:
    """
    Zip up a results folder.
    """
    results_dir = Path(results_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or results_dir.name
    base_name = f"{run_id}-{ts}"
    final_zip = out_dir / f"{base_name}.zip"
    partial_zip = out_dir / f"{base_name}.zip.partial"

    # create archive in a temp location first
    with tempfile.TemporaryDirectory(dir=str(out_dir)) as td:
        tmp_zip_base = Path(td) / base_name  # shutil.make_archive wants a base path w/o extension
        tmp_zip_path = Path(shutil.make_archive(str(tmp_zip_base), "zip", root_dir=str(results_dir)))

        # move into out_dir as .partial then atomically rename to .zip
        shutil.move(str(tmp_zip_path), str(partial_zip))
        os.replace(partial_zip, final_zip)

    return final_zip

configure_path = Path("./config.yml")
config = load_config(path=configure_path)  # CLI argument

# dataname: (slice for training, slice for inference)
datainfo = {
    # our data
    "teaser-gunballoon-dark-acq00002": (slice(60000, 100000), slice(83000, 89000)),
    "bright1": (slice(0, 40000), slice(11300, 15500)),
    "bright2": (slice(0, 40000), slice(2400, 6100)),
    "dark1": (slice(10000, 50000), slice(36500, 41700)),
    "fanclock_bright": (slice(0, 40000), slice(1000, 3000)),
    "fanclock_bright_spadnd": (slice(0, 40000), slice(1000, 3000)),
    "fanclock_dark": (slice(0, 40000), slice(1000, 3000)),
    "teaser_balloonbounce_dark": (slice(10000, 50000), slice(24300, 37900)),
    "teaser_balloonbounce_bright": (slice(20000, 60000), slice(33600, 46500)),
    "teaser-blender-dark": (slice(50000, 90000), slice(70000, 73000)),
    "teaser-blender-bright1": (slice(3000, 43000), slice(39000, 42000)),
    "balloon-laser-acq00000": (slice(0, 40000), slice(5000, 15000)),
    "Feb27_balloonbounce_acq4_3100ppps": (slice(0, 40000), slice(23000, 33000)),
    # b2b data
    "Monkey": (slice(0, 40000), slice(0, 5000)),
    "Resolution_target_drill": (slice(0, 40000), slice(0, 5000)),
}
datanames = [
    "Feb27_balloonbounce_acq4_3100ppps"
]
keep_probs = [1/20]
devices = [1, 2, 3, 4, 5, 6, 7]

FRAME_LIMIT_TRAIN = 40000
FRAME_LIMIT_INF = 10000

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

    slice_interest, slice_interest_inf = datainfo.get(dataname, (None, None))

    for keep_prob in keep_probs:
        # keep_prob = config["PATH"]["thin"]
        print(f"Starting training for {dataname} with keep_prob={keep_prob}")
        if keep_prob < 1.0:
            dcr_prob = prob_from_dcr(dcr_rate_hz=25, fps=100000)
            data = thin_frames_uniform(data_orig, keep_prob=keep_prob, dcr_prob=dcr_prob, seed=42)
        else:
            # copy just in case something funky happens
            data = data_orig.copy()
        if slice_interest is not None:
            data_trainval = data[slice_interest]
            print(f"Using data slice {slice_interest}")
            data_inf = data[slice_interest_inf]
            print(f"Using inference data slice {slice_interest_inf}")
        else:
            data_trainval = data[:FRAME_LIMIT_TRAIN]
            data_inf = data[:FRAME_LIMIT_INF]
        idx_train = int(data_trainval.shape[0] * 0.8)
        traindata = data_trainval[:idx_train]
        valdata = data_trainval[idx_train:]
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
            "persistent_workers": False,  # need this false if looping through multiple model training
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
            devices=devices,
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
        t0 = time.time()
        model.train()
        train_config.to_yaml(default_root_dir / "metadata.yml")
        shutil.copyfile(configure_path, default_root_dir / "config.yml")
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(default_root_dir / "final_model.ckpt")
        t1 = time.time()
        print(f"Training time: {t1 - t0:.2f} seconds")

        del train_loader, val_loader
        del logger
        del trainer
        del model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

        ######## Inference and saving results ########
        model = SPADGAP.load_from_checkpoint(default_root_dir / "final_model.ckpt")
        indata = data_inf.astype(np.float32)
        output = gpu_patch_inference(
            model,
            indata,
            initial_patch_depth=48,
            min_overlap=40,
            device=1,
        )
        resdir = Path(f"results") / test_name
        resdir.mkdir(parents=True, exist_ok=True)
        compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
        zarr.create_array(
            store=resdir / "inference.zarr",
            data=output,
            overwrite=True,
            compressors=compressor,
            chunks=(400, 512, 512),
            attributes={
                "frame_start": slice_interest.start if slice_interest is not None else 0,
                "frame_end": slice_interest.stop if slice_interest is not None else len(data_inf),
            }
        )
        to_video(output, resdir / "inference-gamma.mp4",  playback_fps=30, gamma=1/2.2, cmap="grey", vmin=0)

        del data
        del data_inf
        del train_data, val_data
        del traindata, valdata
        del model
        del indata
        del output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        zip_run(
            resdir,
            "/scratch/jeyan/photon-testing-grounds/output/zipresults",
            run_id=f"b2b-{test_name}"
        )

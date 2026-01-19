# import torch, traceback
# orig = getattr(torch._C, "_cuda_init", None)
# def _wrapped_cuda_init():
#     print("=== torch._C._cuda_init called ===")
#     traceback.print_stack(limit=8)
#     if orig:
#         orig()
# torch._C._cuda_init = _wrapped_cuda_init


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
from tqdm.auto import tqdm


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

keep_prob = 0.031
rel_datapath = Path("data/guitar.bin")
fullmodelname = f"{rel_datapath.stem}-{keep_prob:.3f}"
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

dir_path = rel_datapath
num_of_files = config["PATH"]["num_of_files"]
data_dir = rel_datapath
data_path = ""
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
if keep_prob < 1.0:
    data = thin_frames_uniform(data, keep_prob=keep_prob, seed=42)
idx_train = int(data.shape[0] * 0.8)
data_config = TrainData.from_config(config["DATA"], data[:idx_train].astype(np.float32))
model_config = ModelConfig.from_config(config["MODEL"])
train_config = TrainConfig.from_config(config["TRAINING"], data_config, model_config)
val_data_config = TrainData.from_config_validation(
    config["DATA"], (data[idx_train:].astype(np.float32))
)
print(train_config.metadata())

from matplotlib import cm
import cv2
from tqdm.auto import tqdm
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
def to_video(frames: np.ndarray, path, res_scale=1.0, playback_fps=None, cmap=None, format=None, maxv=None):
    """
    Saves video frame arrays to a video file or sequence of PNGs. If path has no extension, 
    it is treated as a directory and individual image files are saved.

    Args:
        frames (np.ndarray): (T x H x W x C) (RGB) or (T x H x W) (intensity) video frames.
        path (str or Path): output video file path or directory for image files.
        res_scale (float): resolution scaling factor with nearest neighbor interpolation.
        cmap: ignored if frames are RGB; otherwise, matplotlib colormap name or object.
        format (str or None): video format (e.g., "mp4", "avi"), or image format (e.g., "png");
            if None, inferred from path suffix.
    """
    path = Path(path)
    if cmap is None:
        cmap = "viridis"
    cmap_fn = cm.get_cmap(cmap)
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
    if maxv is None:
        maxv = float(np.max(frames))
        if maxv == 0.0:
            maxv = 1.0

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
        if format is None:
            format = "png"
    else:
        if playback_fps is None:
            raise ValueError("playback_fps must be specified if saving a video file")
        if format is None:
            format = path.suffix[1:].lower()
        codec = get_codec_for_format(format)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        vidwriter = cv2.VideoWriter(str(path), fourcc, playback_fps, (out_W, out_H), isColor=True)

    max_frames = len(frames)

    for i in tqdm(range(max_frames), desc="Writing video frames"):
        intensity = np.clip(frames[i], 0, maxv) / maxv  # normalize to [0,1]
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
            frame_path = path / f"frame_{i:05d}.{format}"
            cv2.imwrite(str(frame_path), bgr_mapped)
    if is_video_file:
        vidwriter.release()

from inference import gpu_patch_inference

model = SPADGAP.load_from_checkpoint(f"models/{fullmodelname}/final_model.ckpt")
indata = data[:20000].astype(np.float32)
output = gpu_patch_inference(
    model,
    indata,
    initial_patch_depth=48,
    min_overlap=40,
    device=train_config.device_number,
)
resdir = Path(f"results/{fullmodelname}")
resdir.mkdir(parents=True, exist_ok=True)
imwrite(resdir / "input.tif", indata)
imwrite(resdir / "inference.tif", output)
to_video(output, resdir / "inference.avi",  playback_fps=30, cmap="grey")
gamma = output ** (1/2.2)
gamma14 = output ** (1/4.0)
to_video(gamma, resdir / "inference-gamma.avi",  playback_fps=30, cmap="grey")
to_video(gamma14, resdir / "inference-gamma-1-4.avi",  playback_fps=30, cmap="grey")
output = imread(Path("results") / "inference.tif")

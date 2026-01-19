import numpy as np
from typing import Callable, ClassVar
from collections import defaultdict
from functools import wraps
from PIL.Image import open as open_png
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import yaml

logger = logging.getLogger(__name__)
Loader = Callable[[Path], np.ndarray]
Processer = Callable[[np.ndarray], np.ndarray]
config_path = Path("config.yml")


def load_config(Path: Path) -> dict:
    with open(Path, "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config(config_path)
if "RAW" in config:
    NUM_WORKERS = config["RAW"]["num_workers"]
    IMAGE_SIZE = config["RAW"]["image_size"]
    SUPPORTED_FORMATS = config["RAW"]["supported_formats"]
else:
    NUM_WORKERS = 12
    IMAGE_SIZE = 512
    SUPPORTED_FORMATS = (".png", ".bin", ".npy")


def _try_load(func: Loader) -> Loader:
    """Decorator to catch errors when loading data."""

    @wraps(func)
    def _error_check(path: Path | str, *args, **kwargs) -> np.ndarray:
        try:
            path = Path(path)
            if not path.exists():
                logger.info(f"Path {path} does not exist.")
            output = func(path)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
        return output

    return _error_check


@_try_load
def _bin(path: Path) -> np.ndarray:
    """Loader of binary files."""
    with open(path, "rb") as f:
        raw_data = f.read()
    binframe_length = IMAGE_SIZE * IMAGE_SIZE // 8
    frame_count = len(raw_data) // binframe_length
    bits_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
        frame_count, IMAGE_SIZE, IMAGE_SIZE // 8
    )
    return np.unpackbits(bits_array, axis=2)


@_try_load
def _png(path: Path) -> np.ndarray:
    """Loader of png files using PIL."""
    image_data = np.array(open_png(path))
    if len(image_data.shape) == 3:
        image_data = image_data.transpose(2, 0, 1)
    image_data = image_data[None, ...]

    return image_data


@_try_load
def _npy(path: Path) -> np.ndarray:
    """Loader of npy files using numpy."""
    return np.load(path)


def unbin(path: Path, data: np.ndarray) -> None:
    """Saves a 3D array back to a binary file.

    :param path: Path to the file to be saved.
    :type path: Path
    :param data: Data to be saved.
    :type data: np.ndarray
    """
    packed_bits = np.packbits(data, axis=2)
    flat_data = packed_bits.ravel()
    with open(path, "wb") as f:
        f.write(flat_data.tobytes())


class SPADData:
    """Class to handle SPAD data.

    :param path: Path to the file to be loaded. If None, the object will be created
        without data.
    :type path: Path | None
    :param data: Data to be loaded. If None, the object will be created without data.
    :type data: np.ndarray | None
    """

    _loaders: ClassVar[dict[str, Loader]] = {".bin": _bin, ".png": _png, ".npy": _npy}

    @classmethod
    def register_loader(cls, suffix: str, loader: Loader) -> None:
        """Register a new loader."""
        loader = _try_load(loader)
        cls._loaders[suffix] = loader

    def __init__(self, path: Path | str, loading: bool = False):
        """Class constructor."""
        self.path = Path(path)
        self.name = self.path.stem
        self._data = self.load() if loading else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int | slice) -> np.ndarray:
        return self.data[index]

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return f"SPADData({self.path})"

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self.load()
        return self._data

    def load(self) -> np.ndarray:
        """Load data to the object. Return the loaded data."""
        suffix = self.path.suffix
        if suffix not in SUPPORTED_FORMATS:
            logger.error(f"File type {suffix} is not supported.")
            raise ValueError(f"File type {suffix} is not supported.")
        return self._loaders[suffix](self.path)

    def unload(self):
        """Unload data from the object to free memory."""
        self._data = None

    def __call__(self) -> np.ndarray:
        """Return the data."""
        return self.data

    @property
    def npy_path(self) -> Path:
        """Return the path to save the data."""
        return self.path.with_suffix(".npy")

    def process(self, func: Processer) -> np.ndarray:
        """Process the data with a function."""
        return func(self.data)


class SPADStack:
    """Class to create a stack of SPAD data.

    :param data_list: List of SPADData objects.
    :type data_list: list[SPADData]
    """

    def __init__(self, data_list: list[SPADData]):
        """Class constructor."""
        self.data_list = data_list
        self.reset_stacks()

    def reset_stacks(self):
        self._stack = None
        self._processed_stack = None

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int | slice) -> "SPADStack | SPADData":
        if isinstance(index, slice):
            return SPADStack(self.data_list[index])
        else:
            return self.data_list[index]

    def __str__(self) -> str:
        return "+".join(self.names)

    def __repr__(self) -> str:
        return f"SPADStack({self.names})"

    def __call__(self) -> np.ndarray:
        return self.stack

    @staticmethod
    def _combine_stack(data: list[np.ndarray]) -> np.ndarray:
        """Combine the data into a stack."""
        method = np.concatenate if data[0].ndim >= 3 else np.stack
        return method(data, axis=0)

    @property
    def stack(self) -> np.ndarray:
        if self._stack is None:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                loaded_data = list(executor.map(lambda d: d(), self.data_list))
            self._stack = self._combine_stack(loaded_data)
        return self._stack

    def process(self, func: Processer, concurrent=True) -> np.ndarray:
        """Return the processed data."""
        if self._processed_stack is None:

            def process_spaddata(data: SPADData) -> np.ndarray:
                return data.process(func)

            if concurrent:
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    processed_list = list(
                        executor.map(process_spaddata, self.data_list)
                    )
            else:
                processed_list = [process_spaddata(d) for d in self.data_list]

            self._processed_stack = self._combine_stack(processed_list)
        return self._processed_stack

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.data_list]

    @property
    def name_str(self) -> str:
        return "+".join(self.names)


class SPADFolder:
    """Class to handle a folder of SPAD data.

    :param directory_path: Path to the folder.
    :type directory_path: Path
    """

    def __init__(self, directory_path: Path | str, func: Processer | None = None):
        """Class constructor"""
        self.directory_path = Path(directory_path)
        self.name = self.directory_path.stem
        files = self.directory_path.iterdir()
        self.file_list = defaultdict(list[Path])
        for f in files:
            self.file_list[f.suffix].append(f)
        for key in self.file_list:
            self.file_list[key] = sorted(self.file_list[key])
        self._spadstack = None

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return f"SPADFolder({self.directory_path})"

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, index: int | slice) -> SPADData | list[SPADData]:
        path = self.path_list[index]
        if isinstance(path, list):
            return [SPADData(p) for p in path]
        else:
            return SPADData(path)

    @property
    def file_type(self) -> str:
        return max(self.file_list, key=lambda k: len(self.file_list[k]))

    @property
    def path_list(self, type_: str = "most") -> list[Path]:
        if type_ == "most":
            type_ = self.file_type
        if type_ in self.file_list:
            return self.file_list[type_]
        else:
            logger.error(f"File type {type_} is not supported.")
            raise ValueError(f"File type {type_} is not supported.")

    @property
    def spadstack(self) -> SPADStack:
        """Load the data from the folder."""
        if self._spadstack is None:
            data_list = [SPADData(f) for f in self.path_list]
            self._spadstack = SPADStack(data_list)
        return self._spadstack

    @property
    def data(self) -> np.ndarray:
        return self.spadstack.stack

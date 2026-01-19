import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from numpy.random import randint, rand, seed
from typing import Callable
from metadata import TrainData
from torchvision import transforms
import torchvision.transforms.v2.functional as tf
from torch.distributions.binomial import Binomial
from abc import abstractmethod
import logging

EPSILON = 1e-12

logger = logging.getLogger(__name__)


def lucky() -> bool:
    return rand() < 0.5


class GAPDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray | tuple[np.ndarray, np.ndarray],
        window_size: tuple,
        ground_truth: np.ndarray | None = None,
        virtual_size: int = 0,
        random_crop: bool = False,
        rotation: int = 0,
        augments: bool = False,
        skip_frames: int = 1,
        pileup_correction: int = 0,
        **kwargs,
    ):
        self.data = data if isinstance(data, np.ndarray) else data[0]
        self.target = data[1] if isinstance(data, tuple) else None
        self.ground_truth = ground_truth
        self.window_size = window_size
        self.virtual_size = virtual_size
        self.augments = augments
        self.skip_frames = skip_frames
        self._length = None
        self.rotation = rotation
        self.rotate = transforms.RandomRotation(rotation)
        self.crop = (
            transforms.RandomCrop((self.x_size, self.y_size))
            if random_crop
            else transforms.CenterCrop((self.x_size, self.y_size))
        )
        self.pileup_correction = pileup_correction
        self.kwargs = kwargs
        seed()
        self.__post_init__(**kwargs)

    def __len__(self) -> int:
        return self.length

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, input: np.ndarray):
        self._data = self._convert_to_tensor(input)

    @property
    def target(self) -> torch.Tensor | None:
        return self._target

    @target.setter
    def target(self, input: np.ndarray | None):
        self._target = self._convert_to_tensor(input) if input is not None else None

    @property
    def ground_truth(self) -> torch.Tensor | None:
        return self._ground_truth

    @ground_truth.setter
    def ground_truth(self, input: np.ndarray | None) -> None:
        if input is not None:
            if input.shape != self.data.shape:
                logger.error("Ground truth and data shape must be the same")
                raise ValueError("Ground truth and data shape must be the same")
            self._ground_truth = self._convert_to_tensor(input)
        else:
            self._ground_truth = None

    def _convert_to_tensor(self, input: np.ndarray) -> torch.Tensor:
        try:
            return torch.from_numpy(input)
        except TypeError:
            logger.warning("Data type not supported")
            try:
                logger.info("Trying to convert to int64")
                return torch.from_numpy(input.astype(np.int64))
            except TypeError:
                logger.error("Data type not supported")
                raise TypeError("Data type not supported")

    @property
    @abstractmethod
    def data_size(self) -> int: ...

    @property
    def x_size(self) -> int:
        return self.window_size[-2]

    @property
    def y_size(self) -> int:
        return self.window_size[-1]

    @property
    def z_size(self) -> int:
        return self.window_size[-3] if len(self.window_size) == 3 else 1

    @property
    def ndim(self) -> int:
        return len(self.window_size)

    @property
    def length(self) -> int:
        if self._length is None:
            self._length = (
                self.data_size if self.virtual_size == 0 else self.virtual_size
            )
            if self.skip_frames > 1:
                self._length = self._length // self.skip_frames
        return self._length

    def _crop(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Crop the input data to the window size."""
        if isinstance(self.crop, transforms.CenterCrop):
            return [tf.center_crop(data, [self.x_size, self.y_size]) for data in input]
        else:
            i, j, w, h = self.crop.get_params(input[0], (self.x_size, self.y_size))
            return [tf.crop(data, i, j, w, h) for data in input]

    def _rotate_list(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Rotate the input data if the rotation flag is set to True."""
        if self.rotation > 0:
            angle = rand() * self.rotation
            return [tf.rotate(data, angle) for data in input]
        else:
            return input

    def _rotate_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Rotate the input data if the rotation flag is set to True."""
        if self.rotation > 0:
            return self.rotate(input)
        else:
            return input

    def _aug_output(self, input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply the augmentation to the output data if the augment flag is set to True."""
        if self.augments:
            if lucky():
                input = [torch.transpose(data, -1, -2) for data in input]
            if lucky():
                input = [torch.flip(data, [-1]) for data in input]
            if lucky():
                input = [torch.flip(data, [-2]) for data in input]
        return input

    def _index(self, index: int) -> int:
        index = index if self.virtual_size == 0 else randint(self.data_size)
        if self.skip_frames > 1:
            index = index // self.skip_frames * self.skip_frames
        return index

    def __post_init__(self, **kwargs) -> None:
        """Initialize any other parameters that are needed for the dataset."""
        pass

    @staticmethod
    def _pile_up(input: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        output = F.avg_pool3d(
            input,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        output = output * output * input
        second_photon = torch.bernoulli(output)
        return input + second_photon

    @staticmethod
    def _add_channel_dim(input: list[torch.Tensor]) -> list[torch.Tensor]:
        """Add the channel dimension to the input data."""
        if len(input[0].shape) == 3:
            return [data.unsqueeze(0) for data in input]
        else:
            return [data.swapaxes(0, 1) for data in input]

    @staticmethod
    def xyz_to_window(xy_size: int, z_size: int) -> tuple:
        """Convert the xyz_size to the window tuple."""
        if z_size == 1:
            return (xy_size, xy_size)
        else:
            return (z_size, xy_size, xy_size)

    @classmethod
    def from_config(
        cls, config: dict, data: np.ndarray, ground_truth: np.ndarray | None = None
    ) -> "GAPDataset":
        window_size = cls.xyz_to_window(config["xy_size"], config["z_size"])
        config["data"] = data
        config["ground_truth"] = ground_truth
        config["window_size"] = window_size
        return cls(**config)

    @classmethod
    def from_dataclass(cls, metadata: TrainData) -> "GAPDataset":
        window_size = cls.xyz_to_window(metadata.xy_size, metadata.z_size)
        config = metadata.metadata()
        config["data"] = metadata.data
        config["ground_truth"] = metadata.ground_truth
        config["window_size"] = window_size
        return cls(**config)


class PairedDataset(GAPDataset):
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        index = self._index(index)
        input = self.data[index : index + self.z_size]
        target = (
            self.target[index : index + self.z_size]
            if self.target is not None
            else input
        )
        target = target / (target.mean() + EPSILON)

        if self.ground_truth is not None:
            ground_truth = self.ground_truth[index : index + self.z_size]
            output = self._crop(self._rotate_list([target, input, ground_truth]))
        else:
            output = self._crop(self._rotate_list([target, input]))

        return self._add_channel_dim(self._aug_output(output))

    @property
    def data_size(self) -> int:
        return self.data.shape[0] - self.z_size


class BinomDataset(GAPDataset):
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        index = self._index(index)
        input = self.data[index : index + self.z_size]
        # Combine the input and ground truth data for cropping
        if self.ground_truth is not None:
            ground_truth = self.ground_truth[index : index + self.z_size]
            output = self._crop(self._rotate_list([input, ground_truth]))
            target, noise = self._split_noise(output[0])
            output = [target, noise, output[1]]
        else:
            input = self.crop(self._rotate_tensor(input))
            target, noise = self._split_noise(input)
            output = [target, noise]

        return self._add_channel_dim(self._aug_output(output))

    @property
    def data_size(self) -> int:
        return self.data.shape[0]

    def _split_noise(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the input image into the image and noise components using the distribution"""
        level = self._sample_psnr(input)
        noise = self._sample_noise(input, level)
        target = (input - noise).float()
        target = target / (target.mean() + EPSILON)
        return target, noise.float()

    def _sample_psnr(self, input: torch.Tensor) -> float:
        """Random sample the PSNR level for the input image"""
        # User can pass a custom function to sample the PSNR level
        sample_psnr: Callable | None = self.kwargs.get("sample_psnr")
        if sample_psnr is not None:
            return sample_psnr(input, **self.kwargs)
        else:
            psnr_sampling = self.kwargs.get("psnr_sampling", "db")
            if psnr_sampling == "db":
                return self._sample_db(input)
            elif psnr_sampling == "signal":
                return self._sample_signal()
            elif psnr_sampling == "fixed":
                return self._sample_fixed()
            elif psnr_sampling == "choice":
                return self._sample_choice()
            else:
                logger.warning("PSNR sampling method not supported, using default")
                return self._sample_db(input)

    def _sample_db(self, input: torch.Tensor) -> float:
        """Random sample the PSNR level for the input image"""
        min_psnr = self.kwargs.get("min_psnr", -40.0)
        max_psnr = self.kwargs.get("max_psnr", 40.0)
        max_probability = self.kwargs.get("max_probability", 1 - EPSILON)
        uniform = rand() * (max_psnr - min_psnr) + min_psnr
        level = (10 ** (uniform / 10.0)) / (input.float().mean().item() + EPSILON)
        return max(min(level, max_probability), 1 - max_probability)

    def _sample_signal(self) -> float:
        """Random sample the signal level for the input image"""
        max_probability = self.kwargs.get("max_probability", 1 - EPSILON)
        min_psnr = self.kwargs.get("min_psnr", 0)
        max_psnr = self.kwargs.get("max_psnr", 1 - EPSILON)
        # check min_psnr and max_psnr between 0-1
        if min_psnr < 0 or min_psnr > 1:
            logger.warning("Min PSNR must be between 0 and 1, use 0")
            min_psnr = 0
        if max_psnr < 0 or max_psnr > 1:
            logger.warning(f"Max PSNR must be between 0 and 1, use {max_probability}")
            max_psnr = max_probability
        min_p, max_p = min(min_psnr, max_psnr), max(min_psnr, max_psnr)
        offset = min_p
        scale = max_p - min_p
        level = rand() * scale + offset
        return level

    def _sample_fixed(self) -> float:
        """Random sample the fixed level for the input image"""
        probability = self.kwargs.get("max_probability", 0.5)
        if probability < 0 or probability > 1:
            logger.warning("Probability must be between 0 and 1, using 0.5")
            probability = 0.5
        return probability

    def _sample_choice(self) -> float:
        """Random sample the choice level for the input image"""
        max_probability = self.kwargs.get("max_probability", 1 - EPSILON)
        min_psnr = self.kwargs.get("min_psnr", 0)
        max_psnr = self.kwargs.get("max_psnr", 1 - EPSILON)
        if min_psnr < 0 or min_psnr > 1:
            logger.warning("Min PSNR must be between 0 and 1, use 0")
            min_psnr = 0
        if max_psnr < 0 or max_psnr > 1:
            logger.warning(f"Max PSNR must be between 0 and 1, use {max_probability}")
            max_psnr = max_probability
        return min_psnr if lucky() else max_psnr

    @staticmethod
    def _sample_noise(input: torch.Tensor, level: float) -> torch.Tensor:
        """Sample the noise data for the input image using a binomial distribution"""
        binom = Binomial(total_count=input, probs=torch.tensor([level]))  # type: ignore
        return binom.sample()


class BinomDataset3D(BinomDataset):
    @property
    def data_size(self) -> int:
        return self.data.shape[0] - self.z_size


class ValidationDataset3D(BinomDataset3D):
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        index = self._index(index)
        input = self.data[index : index + self.z_size]
        # Combine the input and ground truth data for cropping
        if self.ground_truth is not None:
            ground_truth = self.ground_truth[index : index + self.z_size]
            output = self._crop([input, input, ground_truth])
        else:
            input = self.crop(input)
            output = [input / (input.mean() + EPSILON), input]

        return self._add_channel_dim(self._aug_output(output))


class BernoulliDataset3D(BinomDataset3D):
    @staticmethod
    def _sample_noise(input: torch.Tensor, level: float) -> torch.Tensor:
        return torch.bernoulli(input * level)


class N2NDataset3D(GAPDataset):
    def __getitem__(self, index: int) -> list[torch.Tensor]:
        start_index = self._index(index)
        end_index = start_index + self.z_size * 2
        output = self.data[start_index:end_index]
        out_even = output[::2].float()
        out_odd = output[1::2].float()
        if self.ground_truth is not None:
            ground_truth = self.ground_truth[start_index:end_index:2].float()
            output = [out_odd, out_even, ground_truth]
        else:
            output = [out_odd, out_even]
        output = self._crop(output)
        return self._add_channel_dim(self._aug_output(output))

    @property
    def data_size(self) -> int:
        return self.data.shape[0] - self.z_size * 2

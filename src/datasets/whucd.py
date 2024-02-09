import glob
import os
from typing import Callable, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchgeo
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datamodules.utils import dataset_split
from torchgeo.datasets.utils import percentile_normalization
from torchgeo.transforms import AugmentationSequential
from torchgeo.transforms.transforms import _ExtractPatches


class WHUCD(torch.utils.data.Dataset):
    splits = ["train", "test"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
    ) -> None:
        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.files = self._load_files(self.root, self.split)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        files = self.files[index]
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
        mask = self._load_target(files["mask"])
        sample = {"image1": image1, "image2": image2, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: str) -> Tensor:
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            tensor = torch.from_numpy(array)
            tensor = tensor.float()
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, path: str) -> Tensor:
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            tensor = torch.from_numpy(array)
            tensor = torch.clamp(tensor, min=0, max=1)
            tensor = tensor.to(torch.long)
            return tensor

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        ncols = 3

        image1 = sample["image1"].permute(1, 2, 0).numpy()
        image1 = percentile_normalization(image1, lower=0, upper=98, axis=(0, 1))

        image2 = sample["image2"].permute(1, 2, 0).numpy()
        image2 = percentile_normalization(image2, lower=0, upper=98, axis=(0, 1))

        if "prediction" in sample:
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 5))

        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")
        axs[2].imshow(sample["mask"], cmap="gray", interpolation="none")
        axs[2].axis("off")

        if "prediction" in sample:
            axs[3].imshow(sample["prediction"], cmap="gray", interpolation="none")
            axs[3].axis("off")
            if show_titles:
                axs[3].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image 1")
            axs[1].set_title("Image 2")
            axs[2].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def _load_files(self, root: str, split: str) -> list[dict[str, str]]:
        images1 = sorted(glob.glob(os.path.join(root, "2012", split, "*.tif")))
        images2 = sorted(glob.glob(os.path.join(root, "2016", split, "*.tif")))
        masks = sorted(glob.glob(os.path.join(root, "change_label", split, "*.tif")))

        files = []
        for image1, image2, mask in zip(images1, images2, masks):
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files


class WHUCDDataModule(NonGeoDataModule):
    def __init__(
        self, patch_size: int = 256, val_split_pct: float = 0.1, *args, **kwargs
    ):
        super().__init__(WHUCD, *args, **kwargs)

        self.patch_size = (patch_size, patch_size)
        self.val_split_pct = val_split_pct

        self.train_aug = AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop(
                size=self.patch_size, scale=(0.8, 1.0), ratio=(1, 1), p=1.0
            ),
            K.Normalize(mean=0.0, std=255.0),
            K.Normalize(mean=0.5, std=0.5),
            data_keys=["image1", "image2", "mask"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0),
            K.Normalize(mean=0.5, std=0.5),
            _ExtractPatches(window_size=self.patch_size),
            data_keys=["image1", "image2", "mask"],
            same_on_batch=True,
        )
        self.test_aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0),
            K.Normalize(mean=0.5, std=0.5),
            _ExtractPatches(window_size=self.patch_size),
            data_keys=["image1", "image2", "mask"],
            same_on_batch=True,
        )

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.dataset = WHUCD(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset = dataset_split(
                self.dataset, val_pct=self.val_split_pct
            )
        if stage == "test":
            self.test_dataset = WHUCD(split="test", **self.kwargs)

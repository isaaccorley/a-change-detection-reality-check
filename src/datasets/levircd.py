import kornia.augmentation as K
import torchgeo.datamodules
import torchgeo.datasets
from torchgeo.transforms import AugmentationSequential
from torchgeo.transforms.transforms import _ExtractPatches


class LEVIRCDDataModule(torchgeo.datamodules.LEVIRCDDataModule):
    train_root = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if stage in ["fit"]:
            self.train_dataset = torchgeo.datasets.LEVIRCD(
                root=self.train_root, split="train"
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = torchgeo.datasets.LEVIRCD(split="val", **self.kwargs)
        if stage == "test":
            self.test_dataset = torchgeo.datasets.LEVIRCD(split="test", **self.kwargs)

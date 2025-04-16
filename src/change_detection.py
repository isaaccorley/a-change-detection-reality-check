# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for change detection."""

import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.models import FCSiamConc, FCSiamDiff, get_weight
from torchgeo.trainers import utils
from torchgeo.trainers.base import BaseTask
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, FBetaScore, JaccardIndex, Precision, Recall
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum

from .models import BIT, ChangeFormerV6, TinyCD


class ChangeDetectionTask(BaseTask):
    """Change Detection."""

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        num_classes: int = 2,
        class_weights: Tensor | None = None,
        labels: list[str] | None = None,
        loss: str = "ce",
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        if ignore_index is not None and loss == "jaccard":
            warnings.warn("ignore_index has no effect on training when loss='jaccard'", UserWarning, stacklevel=2)

        self.weights = weights
        super().__init__(ignore="weights")

    def configure_losses(self) -> None:
        """Initialize the loss criterion.
        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        if loss == "ce":
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_value, weight=self.hparams["class_weights"])
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(mode="multiclass", classes=self.hparams["num_classes"])
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss("multiclass", ignore_index=ignore_index, normalized=True)
        else:
            raise ValueError(f"Loss type '{loss}' is not valid. Currently, supports 'ce', 'jaccard' or 'focal' loss.")

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = self.hparams["ignore_index"]
        labels: list[str] | None = self.hparams["labels"]

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass", num_classes=num_classes, average="micro", multidim_average="global"
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass", num_classes=num_classes, beta=1.0, average="micro", multidim_average="global"
                ),
                "OverallIoU": JaccardIndex(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
                "AverageAccuracy": Accuracy(
                    task="multiclass", num_classes=num_classes, average="macro", multidim_average="global"
                ),
                "AverageF1Score": FBetaScore(
                    task="multiclass", num_classes=num_classes, beta=1.0, average="macro", multidim_average="global"
                ),
                "AverageIoU": JaccardIndex(
                    task="multiclass", num_classes=num_classes, ignore_index=ignore_index, average="macro"
                ),
                "Accuracy": ClasswiseWrapper(
                    Accuracy(task="multiclass", num_classes=num_classes, average="none", multidim_average="global"),
                    labels=labels,
                ),
                "Precision": ClasswiseWrapper(
                    Precision(task="multiclass", num_classes=num_classes, average="none", multidim_average="global"),
                    labels=labels,
                ),
                "Recall": ClasswiseWrapper(
                    Recall(task="multiclass", num_classes=num_classes, average="none", multidim_average="global"),
                    labels=labels,
                ),
                "F1Score": ClasswiseWrapper(
                    FBetaScore(
                        task="multiclass", num_classes=num_classes, beta=1.0, average="none", multidim_average="global"
                    ),
                    labels=labels,
                ),
                "IoU": ClasswiseWrapper(
                    JaccardIndex(task="multiclass", num_classes=num_classes, average="none"), labels=labels
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_optimizers(self):
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(self.trainer.max_epochs + 1)
            return lr_l

        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["lr"], momentum=0.9, weight_decay=5e-4)
        scheduler = scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def configure_models(self) -> None:
        """Initialize the model.
        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels * 2,  # images are concatenated
                classes=num_classes,
            )
        elif model == "fcsiamdiff":
            self.model = FCSiamDiff(
                encoder_name=backbone,
                in_channels=in_channels,
                classes=num_classes,
                encoder_weights="imagenet" if weights is True else None,
            )
        elif model == "fcsiamconc":
            self.model = FCSiamConc(
                encoder_name=backbone,
                in_channels=in_channels,
                classes=num_classes,
                encoder_weights="imagenet" if weights is True else None,
            )
        elif model == "bit":
            self.model = BIT(arch="base_transformer_pos_s4_dd8")
        elif model == "changeformer":
            self.model = ChangeFormerV6(
                input_nc=in_channels, output_nc=num_classes, decoder_softmax=False, embed_dim=256
            )
        elif model == "tinycd":
            self.model = TinyCD(
                bkbn_name="efficientnet_b4", pretrained=True, output_layer_bkbn="3", freeze_backbone=False
            )
        else:
            raise ValueError(f"Model type '{model}' is not valid.")

        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        image1, image2, y = batch["image1"], batch["image2"], batch["mask"]

        model: str = self.hparams["model"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
        elif model in ["fcsiamdiff", "fcsiamconc"]:
            x = torch.stack((image1, image2), dim=1)

        if model in ["bit", "changeformer", "tinycd"]:
            y_hat = self(image1, image2)
        else:
            y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss)

        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)

        self.train_metrics(y_hat_hard, y)
        self.log_dict({f"{k}": v for k, v in self.train_metrics.compute().items()})

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        image1, image2, y = batch["image1"], batch["image2"], batch["mask"]

        model: str = self.hparams["model"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
        elif model in ["fcsiamdiff", "fcsiamconc"]:
            x = torch.stack((image1, image2), dim=1)

        if model in ["bit", "changeformer", "tinycd"]:
            y_hat = self(image1, image2)
        else:
            y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_epoch=True)

        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)

        self.val_metrics(y_hat_hard, y)
        self.log_dict({f"{k}": v for k, v in self.val_metrics.compute().items()}, on_epoch=True)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat_hard
            for key in ["image1", "image2", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                plt.close()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        image1, image2, y = batch["image1"], batch["image2"], batch["mask"]

        model: str = self.hparams["model"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
        elif model in ["fcsiamdiff", "fcsiamconc"]:
            x = torch.stack((image1, image2), dim=1)

        if model in ["bit", "changeformer", "tinycd"]:
            y_hat = self(image1, image2)
        else:
            y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)

        self.log("test_loss", loss, on_epoch=True)

        y_hat = torch.softmax(y_hat, dim=1)
        y_hat_hard = y_hat.argmax(dim=1)

        self.test_metrics(y_hat_hard, y)
        self.log_dict({f"{k}": v for k, v in self.test_metrics.compute().items()}, on_epoch=True)

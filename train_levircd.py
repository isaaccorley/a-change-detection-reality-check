import argparse

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from src.change_detection import ChangeDetectionTask
from src.datasets.levircd import LEVIRCDDataModule


def main(args):
    for seed in range(args.num_seeds):
        lightning.seed_everything(seed)
        datamodule = LEVIRCDDataModule(
            root=args.root, batch_size=args.batch_size, patch_size=256, num_workers=args.workers
        )
        datamodule.train_root = args.train_root
        module = ChangeDetectionTask(
            model=args.model, backbone=args.backbone, weights=True, in_channels=3, num_classes=2, loss="ce", lr=args.lr
        )

        callbacks = ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1)
        trainer = lightning.Trainer(
            accelerator=args.accelerator,
            devices=[args.device],
            logger=True,
            precision="16-mixed",
            max_epochs=args.epochs,
            log_every_n_steps=10,
            default_root_dir=f"logs-levircd-{args.model}",
            callbacks=[callbacks],
        )
        trainer.fit(model=module, datamodule=datamodule)
        trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/levircd")
    parser.add_argument("--train-root", type=str, default="./data/levircd-train-chipped")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "fcsiamconc", "fcsiamdiff", "changeformer", "tinycd", "bit"],
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="only works with unet, fcsiamdiff, or fcsiamconc"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=10)
    args = parser.parse_args()
    main(args)

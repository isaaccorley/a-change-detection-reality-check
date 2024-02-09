import argparse
import glob
import os

import lightning
import pandas as pd
from tqdm import tqdm

from src.change_detection import ChangeDetectionTask
from src.datasets.whucd import WHUCDDataModule


def main(args):
    lightning.seed_everything(0)
    checkpoints = glob.glob(f"{args.ckpt_root}/**/checkpoints/epoch*.ckpt")
    runs = [ckpt.split(os.sep)[-3] for ckpt in checkpoints]

    metrics = {}
    for run, ckpt in tqdm(zip(runs, checkpoints), total=len(runs)):
        datamodule = WHUCDDataModule(
            root=args.root,
            batch_size=args.batch_size,
            patch_size=256,
            num_workers=args.workers,
        )
        module = ChangeDetectionTask.load_from_checkpoint(ckpt, map_location="cpu")
        trainer = lightning.Trainer(
            accelerator=args.accelerator,
            devices=[args.device],
            logger=False,
            precision="16-mixed",
        )
        metrics[run] = trainer.test(model=module, datamodule=datamodule)[0]
        metrics[run]["model"] = module.hparams.model

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.to_csv(args.output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/workspace/storage/data/whucd-chipped")
    parser.add_argument("--ckpt-root", type=str, default="lightning_logs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-filename", type=str, default="metrics.csv")
    args = parser.parse_args()
    main(args)

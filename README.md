# A Change Detection Reality Check

Code and experiments for the paper, ["A Change Detection Reality Check", Isaac Corley, Caleb Robinson, Anthony Ortiz](https://arxiv.org/abs/2402.06994).

### News

- **Model Checkpoints uploaded to HuggingFace [here](https://huggingface.co/isaaccorley/a-change-detection-reality-check)! (see below for checkpoints and metrics)**
- **Our paper has been accepted to the [ICLR 2024 Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2024/)!**

### Summary

Remote sensing image literature from the past several years has exploded with proposed deep learning architectures that claim to be the latest state-of-the-art on standard change detection benchmark datasets. However, has the field truly made significant progress? In this paper we perform experiments which conclude a simple U-Net segmentation baseline without training tricks or complicated architectural changes is still a top performer for the task of change detection.

### Results

We find that U-Net is still a top performer on the LEVIR-CD and WHU-CD benchmark datasets. See below tables for comparisons with SOTA methods.

<p align="center">
    <img src="./assets/levircd-results.png" width="500"/><br/>
    <b>Table 1.</b> Comparison of state-of-the-art and change detection architectures to a U-Net baseline on the LEVIR-CD dataset. We report the test set precision, recall, and F1 metrics of the positive change class. For the baseline experiments we perform 10 runs while varying random the seed and report metrics from the highest performing run. All other metrics are taken from their respective papers. The top performing methods are highlighted in bold. Gray rows indicate our baseline U-Net and siamese encoder variants.
</p>

<p align="center">
    <img src="./assets/whucd-results.png" width="500"/><br/>
    <b>Table 2.</b> Experimental results on the WHU-CD dataset. We retrain several state-of-the-art methods using the original dataset’s train/test splits instead of the commonly used randomly split preprocessed version created in (Bandara & Patel (2022a)). We find that these state-of-the-art methods are outperformed by a U-Net baseline. We report the test set precision, recall, F1, and IoU metrics of the positive change class. For each run we select the model checkpoint with the lowest validation set loss. We provide metrics averaged over 10 runs with varying random seed as well as the best seed. Gray rows indicate our baseline U-Net and siamese encoder variants.
</p>

### Model Checkpoints

#### LEVIR-CD

|    **Model**   	| **Backbone** 	| **Precision** 	| **Recall** 	| **F1** 	| **IoU** 	| **Checkpoint** 	|
|:--------------:	|:---------------:	|:---------:	|:------:	|:------:	|:------:	|:----------:	|
|      U-Net     	|    ResNet-50    	|   0.9197  	| 0.8795 	| 0.8991 	| 0.8167 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/levir-cd/unet_resnet50.ckpt) 	|
|      U-Net     	| EfficientNet-B4 	|   0.9269  	| 0.8588 	| 0.8915 	| 0.8044 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/levir-cd/unet_efficientnetb4.ckpt) 	|
| U-Net SiamConc 	|    ResNet-50    	|   0.9287  	| 0.8749 	| 0.9010 	| 0.8199 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/levir-cd/unet_siamconc_resnet50.ckpt) 	|
| U-Net SiamDiff 	|    ResNet-50    	|   0.9321  	| 0.8730 	| 0.9015 	| 0.8207 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/levir-cd/unet_siamdiff_resnet50.ckpt) 	|

#### WHU-CD (using official train/test splits)

|    **Model**   	| **Backbone** 	| **Precision** 	| **Recall** 	| **F1** 	| **IoU** 	| **Checkpoint** 	|
|:--------------:	|:---------:	|:---------:	|:------:	|:------:	|:------:	|:----------:	|
| U-Net SiamConc 	| ResNet-50 	|    0.8369 	| 0.8130 	| 0.8217 	| 0.7054 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/whu-cd/unet_siamconc_resnet50.ckpt) 	|
| U-Net SiamDiff 	| ResNet-50 	|    0.8856 	| 0.7741 	| 0.8248 	| 0.7086 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/whu-cd/unet_siamdiff_resnet50.ckpt) 	|
|      U-Net     	| ResNet-50 	|    0.8865 	| 0.7663 	| 0.8200 	| 0.7020 	| [Checkpoint](https://huggingface.co/isaaccorley/a-change-detection-reality-check/resolve/main/whu-cd/unet_resnet50.ckpt) 	|

### Reproducing Results

Download the [LEVIR-CD](https://chenhao.in/LEVIR/) and [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) datasets and then use the following notebooks to chip the datasets into non-overlapping 256x256 patches.

```bash
scripts/preprocess_levircd.ipynb
scripts/preprocess_whucd.ipynb
```

To train UNet on both datasets over 10 random seeds run

```bash
python train_levircd.py --train-root /path/to/preprocessed-dataset/ --model unet --backbone resnet50 --num_seeds 10
python train_whucd.py --train-root /path/to/preprocessed-dataset/ --model unet --backbone resnet50 --num_seeds 10
```

To evaluate a set of checkpoints and save results to a .csv file run:

```bash
python test_levircd.py --root /path/to/preprocessed-dataset/ --ckpt-root lightning_logs/ --output-filename metrics.csv
python test_whucd.py --root /path/to/preprocessed-dataset/ --ckpt-root lightning_logs/ --output-filename metrics.csv
```

### Citation

If this work inspired your change detection research, please consider citing our paper:

```
@article{corley2024change,
  title={A Change Detection Reality Check},
  author={Corley, Isaac and Robinson, Caleb and Ortiz, Anthony},
  journal={arXiv preprint arXiv:2402.06994},
  year={2024}
}
```

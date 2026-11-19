# Pet Segmentation and Breed Classification with U-Net

This project trains one model to do two jobs on the Oxford-IIIT Pet dataset:

1. Segment the pet from the background.
2. Classify the breed (37 breeds).

All models are trained from scratch in PyTorch. No pretrained weights are used.

## Dataset

[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

| Split | Images |
|---|---:|
| Train | 3128 |
| Validation | 552 |
| Test | 3669 |

## Notebooks

| File | What it does |
|---|---|
| `00_dataset_exploration.ipynb` | Looks at the dataset |
| `01_base_unet.ipynb` | Base U-Net |
| `02_attention_unet.ipynb` | Attention U-Net |
| `03_base_unet_3class.ipynb` | Base U-Net with 3 class masks |
| `04_attention_unet_3class.ipynb` | Attention U-Net with 3 class masks |
| `05_base_unet_no_augmentation.ipynb` | Base U-Net without augmentation |
| `06_attention_unet_no_augmentation.ipynb` | Attention U-Net without augmentation |
| `07_densenet_unet.ipynb` | U-Net with DenseNet-121 encoder |
| `08_efficientnet_unet.ipynb` | U-Net with EfficientNet-B0 encoder |
| `09_mobilenetv2_unet.ipynb` | U-Net with MobileNetV2 encoder |
| `10_resnet_unet.ipynb` | U-Net with ResNet-34 encoder |
| `11_base_unet_rmsprop.ipynb` | Base U-Net with RMSProp |
| `12_base_unet_sgd_momentum.ipynb` | Base U-Net with SGD + Momentum |
| `13_attention_unet_rmsprop.ipynb` | Attention U-Net with RMSProp |
| `14_attention_unet_sgd_momentum.ipynb` | Attention U-Net with SGD + Momentum |
| `15_results_comparison.ipynb` | All results in one place |
| `16_inference_demo.ipynb` | Load a trained model and predict on your own image |

## Training setup

- Image size: 224 x 224
- Batch size: 20
- Epochs: 55
- Optimizer: AdamW (unless the notebook name says otherwise)
- Loss: cross entropy for breed + cross entropy and Dice for the mask
- Augmentation: RandAugment, color jitter, random erasing, MixUp
- GPU: Kaggle Tesla T4

## Results on the test set

| Model | mIoU | Dice | Breed accuracy | F1 |
|---|---:|---:|---:|---:|
| Base U-Net | 0.8282 | 0.9058 | 0.7130 | 0.7063 |
| Attention U-Net | 0.8322 | 0.9082 | 0.7274 | 0.7195 |
| Base U-Net (3 class) | 0.6992 | 0.8067 | 0.7389 | 0.7318 |
| Attention U-Net (3 class) | 0.7014 | 0.8083 | 0.7403 | 0.7336 |
| Base U-Net (no augmentation) | 0.8935 | 0.9437 | 0.6882 | 0.6862 |
| Attention U-Net (no augmentation) | 0.8950 | 0.9445 | 0.7013 | 0.6984 |
| DenseNet-121 | 0.8141 | 0.8973 | 0.6639 | 0.6591 |
| EfficientNet-B0 | 0.7778 | 0.8745 | 0.5647 | 0.5573 |
| MobileNetV2 | 0.7509 | 0.8573 | 0.4192 | 0.3955 |
| ResNet-34 | 0.7864 | 0.8800 | 0.6236 | 0.6199 |
| Base U-Net (RMSProp) | 0.7557 | 0.8603 | 0.3729 | 0.3525 |
| Base U-Net (SGD + Momentum) | 0.8206 | 0.9012 | 0.7389 | 0.7337 |
| Attention U-Net (RMSProp) | 0.7634 | 0.8654 | 0.4336 | 0.4156 |
| Attention U-Net (SGD + Momentum) | 0.8247 | 0.9037 | 0.7310 | 0.7270 |

Train and validation scores are in `15_results_comparison.ipynb`.

## How to run

1. Open a notebook in Kaggle.
2. Turn on the GPU (T4).
3. Add the Oxford-IIIT Pet dataset as input. If you skip this, turn on Internet and the notebook will download it.
4. Run all cells.

Each training notebook takes about 1 to 4 hours.


## Requirements

Python 3, PyTorch, torchvision, scikit-learn, pandas, matplotlib, Pillow, tqdm.

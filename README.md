



# Watermark Dataset Generator & Detector

**This is the pilot project to verify the feasibility of the overall approach.**

This project generates paired datasets of original and watermarked images using [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and a custom decoder trained with the Stable Signature watermarking method.

The generated dataset will later be used to train a discriminator to distinguish between original and watermarked images.

## 🖼️ Overview

Given COCO caption annotations, we use them as prompts to generate images with two decoding strategies:

- **Original**: Uses the original Stable Diffusion decoder.
- **Watermarked**: Replaces the decoder with a fine-tuned version ( Stable Signature decoder) to embed an invisible watermark.

 We generate all images with fixed seed.

## 🚀 Usage

1. Clone repository:

```
git clone https://github.com/yourusername/watermark-dataset-generator.git
cd watermark-dataset-generator
```

2. Install dependencies:

```python
conda env create -f environment.yml -n forgery-watermark
pip install -r requirements.txt
```

3. Prepare model checkpoints:

- Get LDM configs from [Stable Diffusion](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion) repositories
- Download [Stable Diffusion v2.1 checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main)
- Obtain fine-tuned decoder weights ([sd2_decoder.pth](https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth))
- Download COCO Caption 

```python
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

### Basic Generation

```
python generate_dataset.py \
  --output_dir ./results \
  --ldm_config ./configs/v2-inference.yaml \
  --ldm_ckpt ./models/v2-1_512-ema-pruned.ckpt \
  --decoder_ckpt ./models/sd2_decoder.pth \
  --ann_path ./data/coco/annotations/captions_train2014.json \
  --batch_size 8 \
  --num_samples 1000 \
  --seed 42
```

### Arguments

| Parameter        | Default                                       | Description                 |
| :--------------- | :-------------------------------------------- | :-------------------------- |
| `--output_dir`   | ./results                                     | Output directory            |
| `--ldm_config`   | configs/v2-inference.yaml                     | LDM config path             |
| `--ldm_ckpt`     | models/v2-1_512-ema-pruned.ckpt               | LDM checkpoint path         |
| `--decoder_ckpt` | models/sd2_decoder.pth                        | Watermarked decoder weights |
| `--ann_path`     | data/coco/annotations/captions_train2014.json | COCO annotations path       |
| `--batch_size`   | 8                                             | Generation batch size       |
| `--num_samples`  | 16                                            | Total samples to generate   |
| `--seed`         | 42                                            | Random seed                 |

## 📂 Output Structure

```
output_dir/
├── original/
│   ├── img_0000.png
│   ├── img_0001.png
│   └── ...
├── watermarked/
│   ├── img_0000.png
│   ├── img_0001.png
│   └── ...
└── metadata.json
```

`metadata.json` contains:

```
[
  {
    "id": 0,
    "prompt": "A red bicycle parked next to a tree",
    "original": "./results/original/img_0000.png",
    "watermarked": "./results/watermarked/img_0000.png"
  },
  ...
]
```

## 🧠 Training Detector (Upcoming)

Waiting......

## Acknowledgements

https://github.com/facebookresearch/stable_signature
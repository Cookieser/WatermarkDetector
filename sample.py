import os
import torch 
import sys
import numpy as np
sys.path.append('src')
from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from utils_model import load_model_from_config 
import random
import json
import argparse
from pycocotools.coco import COCO


def main(args):
    batch_size = args.batch_size
    output_dir = args.output_dir
    ldm_config = args.ldm_config
    ldm_ckpt = args.ldm_ckpt
    ann_path = args.ann_path
    seed = args.seed
    decoder_path = args.decoder_ckpt
    num_samples = args.num_samples


    device = torch.device("cuda")
    os.makedirs(f"{output_dir}/original", exist_ok=True)
    os.makedirs(f"{output_dir}/watermarked", exist_ok=True)


    # Fix the seed
    
    np.random.seed(seed)
    random.seed(seed)


    coco = COCO(ann_path)

    img_ids = coco.getImgIds()
    random.seed(42)
    random.shuffle(img_ids)


    captions = []
    for img_id in img_ids[:num_samples]:  
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if anns:
            captions.append(anns[0]['caption'])  

    print(f"Total images selected: {len(captions)}")
    print("Example captions:")
    for i, cap in enumerate(captions[:5]):
        print(f"{i+1}: {cap}")


    print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
    config = OmegaConf.load(f"{ldm_config}")
    ldm_ae = load_model_from_config(config, ldm_ckpt)
    ldm_aef = ldm_ae.first_stage_model
    ldm_aef.eval()

    # loading the fine-tuned decoder weights
    state_dict = torch.load(decoder_path)
    unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
    print(unexpected_keys)
    print("you should check that the decoder keys are correctly matched")

    # loading the pipeline, and replacing the decode function of the pipe
    model = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionPipeline.from_pretrained(model).to(device)


    dataset_records = []
    original_decode = pipe.vae.decode
    for idx in range(0, len(captions), batch_size):
        batch_prompts = captions[idx:idx+batch_size]
        generator = torch.Generator(device=device).manual_seed(seed + idx)
        pipe.vae.decode = original_decode  
        image_origs = pipe(batch_prompts, generator=generator).images

        orig_paths = []
        for i, img in enumerate(image_origs):
            orig_path = f"{output_dir}/original/img_{idx + i:04d}.png"
            img.save(orig_path)
            orig_paths.append(orig_path)


        generator = torch.Generator(device=device).manual_seed(seed + idx)
        pipe.vae.decode = lambda x, *args, **kwargs: ldm_aef.decode(x).unsqueeze(0)
        image_watermarked = pipe(batch_prompts, generator=generator).images

        for i, img in enumerate(image_origs):
            wm_path = f"{output_dir}/watermarked/img_{idx + i:04d}.png"
            img.save(wm_path)
        
            dataset_records.append({
                "id": idx + i,
                "prompt": batch_prompts[i],
                "original": orig_paths[i],
                "watermarked": wm_path
            })

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(dataset_records, f, indent=2)

    print(f"✅ Done! Total samples: {len(dataset_records)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate original and watermarked images from COCO captions.")
    parser.add_argument('--output_dir', type=str, default='test', help='Directory to save outputs')
    parser.add_argument('--ldm_config', type=str, default='ckpt/v2-inference.yaml', help='Path to LDM config file')
    parser.add_argument('--ldm_ckpt', type=str, default='ckpt/v2-1_512-ema-pruned.ckpt', help='Path to LDM checkpoint')
    parser.add_argument('--decoder_ckpt', type=str, default='ckpt/sd2_decoder.pth', help='Path to decoder weights')
    parser.add_argument('--ann_path', type=str, default='coco/annotations/captions_train2014.json', help='Path to COCO captions JSON')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of image-caption samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
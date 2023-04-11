import os.path
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusers.schedulers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import wandb
import argparse
from distutils.util import strtobool
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


def image_grid(images, rows=2, cols=2):
    w, h = images[0].size
    grid = Image.new('RGB', size=(w * cols, h * rows))
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,)
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'))
    parser.add_argument('--wandb-project', type=str, default='hands_on_diffusion')
    parser.add_argument('--wandb-entity', type=str, default='Slientea98')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cache-path', type=str, default='/home/yibo/.cache/huggingface/datasets')
    parser.add_argument('--huggan-dataset', type=str, default='huggan/smithsonian_butterflies_subset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,)
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--prompt', type=str, default='', nargs='*')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_id = 'runwayml/stable-diffusion-v1-5'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    prompt = args.prompt
    pipe.to(torch.device(args.device))
    generator = torch.Generator('cuda').manual_seed(42)
    images = pipe(prompt, generator=generator, num_inference_steps=50).images
    image_grid(images, rows=1, cols=1).show()

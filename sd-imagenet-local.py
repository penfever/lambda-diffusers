from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import requests
import json
from torchvision import transforms
import numpy as np

imagenet_path = Path("/data/datasets/ImageNet/train")

imagenet_images = list(imagenet_path.rglob("*.JPEG"))

with open("completed_images.json", "r") as f:
    completed_images = json.load(f)

imagenet_images = list(set(imagenet_images) - set([Path(p) for p in completed_images]))

imagenet_wnids = [p.parts[-2] for p in imagenet_images]

imagenet_fns = [p.parts[-1] for p in imagenet_images]

dest_path = Path("/data/datasets/sd-imagenet")

from lambda_diffusers import StableDiffusionImageEmbedPipeline
from PIL import Image

device = "cuda:0"
sd_pipe = StableDiffusionImageEmbedPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="273115e88df42350019ef4d628265b8c29ef4af5",
    )
sd_pipe = sd_pipe.to(device)

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])

for idx, p in tqdm(enumerate(imagenet_images), total=len(imagenet_images)):
    completed_images.append(str(p))
    if len(completed_images) % 1000 == 0:
        print("Saving completed images list")
        with open("completed_images.json", "w") as f:
            json.dump(completed_images, f, indent=4)
    target_path = dest_path / Path(imagenet_wnids[idx])
    target_path.mkdir(parents=True, exist_ok=True)
    target_file = target_path / Path(imagenet_fns[idx])
    img = Image.open(p)
    try:
        out = sd_pipe(img, guidance_scale=3)
        out["sample"][0].save(target_file)
    except Exception as e:
        print(e)
        continue
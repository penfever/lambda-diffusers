import replicate
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import requests
import json
from torchvision import transforms
import numpy as np
from retry import retry

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

@retry(tries=3, delay=3, backoff=2)
def get_replicate(p):
    output = replicate.run(
        "lambdal/stable-diffusion-image-variation:7c399ba0e1b33ed8ec39ed30eb6b0a2d9e054462543c428c251293034af82a8e",
        input={"input_image": open(p, "rb")}
    )
    return output

for idx, p in tqdm(enumerate(imagenet_images), total=len(imagenet_images)):
    completed_images.append(str(p))
    if len(completed_images) % 1000 == 0:
        print("Saving completed images list")
        with open("completed_images.json", "w") as f:
            json.dump(completed_images, f, indent=4)
    target_path = dest_path / Path(imagenet_wnids[idx])
    target_path.mkdir(parents=True, exist_ok=True)
    target_file = target_path / Path(imagenet_fns[idx])
    try:
        output = get_replicate(p)
        data = requests.get(output[0]).content
        with open(target_file, 'wb+') as f:
            f.write(data)
    except Exception as e:
        print(e)
        continue
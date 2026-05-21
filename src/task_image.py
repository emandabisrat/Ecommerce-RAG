from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


class ProductImageGenerator:

    def __init__(self):

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )

        self.pipe = self.pipe.to(device)

    def generate(
        self,
        prompt: str,
        output_path: str = "generated_product.png",
    ):

        image = self.pipe(prompt).images[0]

        image.save(output_path)

        return output_path
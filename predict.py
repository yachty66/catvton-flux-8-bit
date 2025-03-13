from typing import List

from cog import BasePredictor, Input, Path, Secret
from diffusers.utils import load_image
from diffusers import FluxFillPipeline
from diffusers import FluxTransformer2DModel
import torch
from torchvision import transforms

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load part of the model into memory to make running multiple predictions efficient"""
        self.try_on_transformer = FluxTransformer2DModel.from_pretrained("xiaozaa/catvton-flux-beta", 
            torch_dtype=torch.bfloat16)
        self.try_off_transformer = FluxTransformer2DModel.from_pretrained("xiaozaa/cat-tryoff-flux", 
            torch_dtype=torch.bfloat16)
        
    def predict(self,
                hf_token: Secret = Input(description="Hugging Face API token. Create a write token at https://huggingface.co/settings/token. You also need to approve the Flux Dev terms."),
                image: Path = Input(description="Image file path", default="https://github.com/nftblackmagic/catvton-flux/raw/main/example/person/1.jpg"),
                mask: Path = Input(description="Mask file path", default="https://github.com/nftblackmagic/catvton-flux/blob/main/example/person/1_mask.png?raw=true"),
                try_on: bool = Input(False, description="Try on or try off"),
                garment: Path = Input(description="Garment file path like https://github.com/nftblackmagic/catvton-flux/raw/main/example/garment/00035_00.jpg", default=None),
                num_steps: int = Input(50, description="Number of steps to run the model for"),
                guidance_scale: float = Input(30, description="Guidance scale for the model"),
                seed: int = Input(0, description="Seed for the model"),
                width: int = Input(576, description="Width of the output image"),
                height: int = Input(768, description="Height of the output image"))  -> List[Path]:
                
        size = (width, height)
        i = load_image(str(image)).convert("RGB").resize(size)
        m = load_image(str(mask)).convert("RGB").resize(size)

        if try_on:
            g = load_image(str(garment)).convert("RGB").resize(size)
            self.transformer = self.try_on_transformer
        else:
            self.transformer = self.try_off_transformer

        self.pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=self.transformer,
            torch_dtype=torch.bfloat16,
            token=hf_token.get_secret_value()
        ).to("cuda")

        self.pipe.transformer.to(torch.bfloat16)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # For RGB images
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Transform images using the new preprocessing
        image_tensor = transform(i)
        mask_tensor = mask_transform(m)[:1]  # Take only first channel
        if try_on:
            garment_tensor = transform(g)
        else:
            garment_tensor = torch.zeros_like(image_tensor)
            image_tensor = image_tensor * mask_tensor

        # Create concatenated images
        inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)  # Concatenate along width
        garment_mask = torch.zeros_like(mask_tensor)

        if try_on:
            extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
        else:
            extended_mask = torch.cat([1 - garment_mask, garment_mask], dim=2)

        prompt = f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; " \
                f"[IMAGE1] Detailed product shot of a clothing" \
                f"[IMAGE2] The same cloth is worn by a model in a lifestyle setting."
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            height=size[1],
            width=size[0] * 2,
            image=inpaint_image,
            mask_image=extended_mask,
            num_inference_steps=num_steps,
            generator=generator,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            prompt=prompt,
        ).images[0]

        # Split and save results
        width = size[0]
        garment_result = result.crop((0, 0, width, size[1]))
        try_result = result.crop((width, 0, width * 2, size[1]))
        out_path = "/tmp/try.png"
        try_result.save(out_path)
        garm_out_path = "/tmp/garment.png"
        garment_result.save(garm_out_path)
        return [Path(out_path), Path(garm_out_path)]
            
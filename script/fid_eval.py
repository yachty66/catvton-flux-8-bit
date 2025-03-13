from PIL import Image
import os
import numpy as np
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


# Paths setup
generated_dataset_path = "output/tryon_results"
original_dataset_path = "data/VITON-HD/test/image"  # Replace with your actual original dataset path

# Get generated images
image_paths = sorted([os.path.join(generated_dataset_path, x) for x in os.listdir(generated_dataset_path)])
generated_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

# Get corresponding original images
original_images = []
for gen_path in image_paths:
    # Extract the XXXXXX part from "tryon_XXXXXX.jpg"
    base_name = os.path.basename(gen_path)  # get filename from path
    original_id = base_name.replace("tryon_", "")  # remove "tryon_" prefix
    
    # Construct original image path
    original_path = os.path.join(original_dataset_path, original_id)
    original_images.append(np.array(Image.open(original_path).convert("RGB")))
    


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (768, 1024))

real_images = torch.cat([preprocess_image(image) for image in original_images])
fake_images = torch.cat([preprocess_image(image) for image in generated_images])
print(real_images.shape, fake_images.shape)

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
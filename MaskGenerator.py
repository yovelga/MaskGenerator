import cv2
import torch
import base64
import os
import requests
import shutil
import numpy as np
import supervision as sv
import sys
import tifffile as tiff


def padding_mask(x_min, y_min, x_max, y_max, px_size=10):
    padding = px_size
    xmin = max(0, x_min - padding)
    ymin = max(0, y_min - padding)
    xmax = min(image_w, x_max + padding)
    ymax = min(image_h, y_max + padding)
    return [xmin, ymin, xmax, ymax]

HOME = os.getcwd()
print("HOME:", HOME)

sys.path.append(os.path.abspath(f"{HOME}/segment-anything-2"))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Check CUDA availability
print(torch.__version__)
print("is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA is available:", torch.cuda.get_device_name())
else:
    print("No GPU available")

# Define paths
SAM_weights = f"{HOME}/checkpoints/sam2_hiera_large.pt"
images_path = os.path.join(HOME, "images")

# Ensure directories exist
os.makedirs(images_path, exist_ok=True)

# Download SAM weights if not available
if not os.path.exists(SAM_weights):
    checkpoint_dir = os.path.join(HOME, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    url = "https://example.com/path/to/sam2_hiera_large.pt"  # Replace with actual URL
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(SAM_weights, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Checkpoint downloaded successfully to {SAM_weights}")
    else:
        print(f"Error while downloading: {response.status_code}")
else:
    print("SAM weights already exist. Skipping download.")

# Initialize SAM2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = f"{HOME}/checkpoints/sam2_hiera_large.pt"
CONFIG = f"{HOME}/segment-anything-2/sam2/configs/sam2/sam2_hiera_l.yaml"
sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

# Create the 'MASKS' folder
masks_path = os.path.join(HOME, "masks")
os.makedirs(masks_path, exist_ok=True)

# Process images
images_list = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not images_list:
    print("No images found in the base path. Exiting...")
else:
    for image_name in images_list:
        # Get the image name without extension
        image_name_without_extension = os.path.splitext(image_name)[0]

        # Read and preprocess the image
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape  # Get image dimensions
        image = cv2.resize(image, (512, 512))

        # Define the 'used_data' folder path
        used_images = os.path.join(HOME, "used")

        # Create the 'used_data' folder if it doesn't exist
        os.makedirs(used_images, exist_ok=True)
        shutil.move(image_path, os.path.join(used_images, image_name))

        # Generate detections
        sam2_result = mask_generator.generate(image)
        detections = sv.Detections.from_sam(sam_result=sam2_result)

        # Store detection data
        detections_data = []
        for i in range(len(detections)):
            mask = detections[i].mask[0]  # Extract the mask
            bbox_original = detections[i].xyxy[0]
            x_min, y_min, x_max, y_max = map(int, bbox_original)

            bbox_extra = padding_mask(x_min, y_min, x_max, y_max)

            # Save mask as TIF
            mask_tif_path = os.path.join(masks_path, f"{image_name_without_extension}_mask_{i}.tif")

            metadata = {
                "image_name": image_name,
                "original_bbox": bbox_original.tolist(),
                "padded_bbox": bbox_extra,
                "tag": "none"
            }

            tiff.imwrite(
                mask_tif_path,
                mask,  # Binary mask (0 and 1 only)
                dtype="bool",  # Save as binary format
                compression="LZW",  # Lossless compression
                metadata=metadata  # Add metadata
            )

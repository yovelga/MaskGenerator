import cv2
import torch
import base64
import os
import requests
import shutil
import json
import numpy as np
import supervision as sv
import sys
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
jsons_path = os.path.join(HOME, "JSONs")
base_path = os.path.join(HOME, "images")

# Ensure directories exist
os.makedirs(jsons_path, exist_ok=True)
os.makedirs(base_path, exist_ok=True)

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

# Process images
images_list = [f for f in os.listdir(base_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not images_list:
    print("No images found in the base path. Exiting...")
else:
    for image_name in images_list:
        # Read and preprocess the image
        image_path = os.path.join(base_path, image_name)
        image = cv2.imread(image_path)
        image_h, image_w, _ = image.shape  # Get image dimensions
        image = cv2.resize(image, (512, 512))

        # Create folder for detections of this image
        image_name_without_extension = os.path.splitext(image_name)[0]
        image_detections_folder = os.path.join(jsons_path, image_name_without_extension)
        os.makedirs(image_detections_folder, exist_ok=True)

        # Overwrite the existing file if it exists
        destination_path = os.path.join(image_detections_folder, image_name)
        if os.path.exists(destination_path):
            os.remove(destination_path)  # Remove the existing file

        shutil.copy(image_path, destination_path)  # copy the file to the folder

        # Define the 'used_data' folder path
        used_data_folder = os.path.join(base_path, 'used_data')

        # Create the 'used_data' folder if it doesn't exist
        os.makedirs(used_data_folder, exist_ok=True)
        shutil.move(image_path, os.path.join(image_path, used_data_folder))


        # Generate detections
        sam2_result = mask_generator.generate(image)
        detections = sv.Detections.from_sam(sam_result=sam2_result)

        # Store detection data
        detections_data = []
        for i in range(len(detections)):
            mask = detections[i].mask[0]  # Extract the mask
            bbox_original = detections[i].xyxy[0]
            x_min, y_min, x_max, y_max = map(int, bbox_original)

            # Expand bounding box with padding
            padding = 10
            xmin = max(0, x_min - padding)
            ymin = max(0, y_min - padding)
            xmax = min(image_w, x_max + padding)
            ymax = min(image_h, y_max + padding)
            bbox_extra = [xmin, ymin, xmax, ymax]

            # Convert mask to base64
            _, mask_encoded = cv2.imencode('.png', mask * 255)
            mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')

            # Add detection to JSON
            detections_data.append({
                "image_name": image_name,
                "image_path": os.path.join(image_detections_folder, image_name),
                "original_bbox": bbox_original.tolist(),
                "expanded_bbox": bbox_extra,
                "mask": mask_base64,
                "tag": "none"
            })

        # Save detections to JSON
        json_file_path = os.path.join(image_detections_folder, f"{image_name_without_extension}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(detections_data, json_file, indent=4)
            print(f"Saved detections for {image_name_without_extension} to {json_file_path}")

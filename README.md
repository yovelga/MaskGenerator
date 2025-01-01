# Project: Object Detection and Mask Generation

## Overview

This project is a critical part of a larger workflow aimed at automating object detection and classification in images. The primary goal of this phase is to detect all objects in an image, generate their corresponding masks, and save these masks for future classification. The classification step, which will follow, will determine the relevance of the detected objects and filter images accordingly.

## Objective

- Identify objects in an image using bounding boxes.
- Generate binary masks for each detected object.
- Store the masks as TIFF files with relevant metadata for future processing.

## Key Features

- **Object Detection:** Detects objects in an image and generates bounding boxes for each.
- **Mask Generation:** Creates binary masks for the detected objects.
- **Metadata Storage:** Stores object-related metadata (e.g., bounding box coordinates) directly within the TIFF files.
- **Preparation for Classification:** Saves masks for subsequent use in a classification pipeline.

## Workflow

1. **Input:**

   - The script processes images from the `images` directory.
   - Supported formats: `.png`, `.jpg`, `.jpeg`.

2. **Object Detection and Mask Creation:**

   - Utilizes the Segment Anything Model (SAM) to detect objects.
   - Generates binary masks for each detected object.

3. **Output:**

   - Saves masks as TIFF files in the `masks` directory.
   - Moves processed images to the `used` directory.

4. **Metadata:**

   - Stores metadata (e.g., bounding box coordinates) in the TIFF file for each mask.

## Directory Structure

```
project/
|-- images/       # Input images
|-- masks/        # Generated masks in TIFF format
|-- used/         # Processed images
|-- checkpoints/  # SAM model weights
|-- segment-anything-2/ # SAM source code
```

## Setup

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the SAM model weights are downloaded to the `checkpoints` directory:
   - **Model weights:** `sam2_hiera_large.pt`
   - **Configuration file:** `sam2_hiera_l.yaml`

## Usage

Run the script to process images:

```bash
python firstTry.py
```

## Future Steps

- Implement object classification to filter relevant objects.
- Optimize the detection pipeline for efficiency.

## Contact

For any questions or feedback, please contact Yovel at [YovelGani@gmail.com](mailto:YovelGani@gmail.com).


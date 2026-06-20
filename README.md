# Synthetic Dataset Generator For YOLO-OBB Detection
## Overview
This program generates synthetic training and validation datasets for OBB (Oriented Bounding Box) detection with various types of interference patterns. It creates images with labeled real targets and unlabeled distractors to simulate real-world detection challenges.
This repository is released under the `GNU GPLv3` license.
## Key Features
1. Generates labeled targets (real objects) placed in four quadrants per image
2. Adds unlabeled false targets (cropped and rotated fragments of real objects)
3. Draws random geometric interruptions (lines, polygons) as visual noise
4. Pastes external false images as additional distractors
5. Applies brightness variations to simulate lighting changes
6. Adds Gaussian blur for motion blur / out-of-focus simulation
7. Optional downscaling for scale variation
8. Released with simple Ultralytics YOLO training scripts

## Directory Structure Required
Before running the program, create the following directories:
```
./background/         - Contains background images (PNG, JPG, JPEG). You can use COCO and DOTA dataset images
./target/train/0/     - Training target images (class 0)
./target/val/0/       - Validation target images (class 0)
./false_images/       - External images for false positives (optional)
```
Note: 
1. For multi-class detection, create subdirectories 0, 1, 2, etc. under both train and val folders.
2. Target images MUST be in **RGBA** format with **transparancy channel**.
3. `rembg.py` in the `toolkit` folder can remove background and create **transparent** PNG files.

Output Directory Structure
The program creates a ./data0/ directory with the following structure:
```
./data0/
   ├── train/
   │   ├── images/     - Generated training images (1.jpg, 2.jpg, ...)
   │   └── labels/     - Corresponding YOLO format labels (1.txt, 2.txt, ...)
   └── val/
       ├── images/     - Generated validation images
       └── labels/     - Corresponding YOLO format labels
```
## Label Format
The program uses YOLO polygon format:
class x1 y1 x2 y2 x3 y3 x4 y4

Where:
- class: Object class ID (0 for single class)
- x1 y1 ... x4 y4: Four corner coordinates of rotated bounding box
- All coordinates are normalized to [0, 1] range (relative to image size)

## Configurable Parameters
Edit the following parameters at the top of the script:
```
DATASET SIZE:
  TRAIN_NUM = 800       - Number of training images
  VAL_NUM = 200         - Number of validation images
  
OBJECT PROPERTIES:
  NC = 1                - Number of object classes
  MIN_SIZE = 32         - Minimum target size (pixels)
  MAX_SIZE = 200        - Maximum target size (pixels)
  IMG_SIZE = 1280       - Image dimensions (width x height)
  BORDER = 160          - Margin from borders for target placement

INTERFERENCE PARAMETERS:
  FALSE_TARGET_PROB = 0.5     - Probability of adding false targets
  FALSE_TARGET_COUNT = 24     - Maximum false targets per image
  INTERRUPTION_PROB = 0.25    - Probability of adding interruptions
  INTERRUPTION_COUNT = 24     - Maximum interruptions per image
  BRIGHTNESS_RANGE = (0.6, 1.6) - Brightness factor range (1.0=original)
  BLUR_PROB = 0.3             - Probability of applying blur
  BLUR_RADIUS_RANGE = (0.3, 1.0) - Gaussian blur radius range
  RESIZE_PROB = 0.3           - Probability of downscaling
  RESIZE_SIZE = 640           - Target size when downscaled
  FALSE_IMAGE_PROB = 0.5      - Probability of adding external false images
  FALSE_IMAGE_COUNT = 24      - Maximum false images per image

INTERRUPTION DRAWING:
  LINE_WIDTH_RANGE = (2, 8)      - Line thickness range (pixels)
  POLYGON_SIDES_RANGE = (2, 16)  - Number of polygon vertices range

random.seed(1337)             - Random seed

```
## Run The Program
### Requirements
------------
Python packages:
  - Pillow (PIL)
  - numpy

Install with:
  `pip install Pillow numpy`

### RUNNING THE PROGRAM
1. Ensure all required directories exist with appropriate images
2. Run the script:
   python dataset_generator_obb.py
3. Monitor progress in console output

### File Naming
Images are saved as 1.jpg, 2.jpg, etc.
Labels are saved as 1.txt, 2.txt, etc.
The index numbers correspond between images and labels.

### Interference Processing Order
1. Load background image
2. Draw geometric interruptions (lines, polygons)
3. Add false targets (cropped real objects)
4. Paste external false images
5. Place labeled targets (four per image)
6. Apply Gaussian blur (if selected)
7. Apply downscaling (if selected)
8. Save image and label files

## OUTPUT EXAMPLES
The generated dataset is compatible with:
- YOLO-based object detection frameworks
- Any detector that uses polygon bounding boxes (OBB - Oriented Bounding Box)
- Training pipelines that handle rotated objects


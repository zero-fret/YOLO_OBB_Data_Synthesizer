"""
Synthetic Dataset Generator for Object Detection
Creates training and validation datasets with multiple types of interference patterns
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# ==================== Dataset Configuration ====================
TRAIN_NUM = 800          # Number of training images to generate
VAL_NUM = 200             # Number of validation images to generate
NC = 1                    # Number of object classes
MIN_SIZE = 32             # Minimum target size in pixels
MAX_SIZE = 200            # Maximum target size in pixels

# Image dimensions
IMG_SIZE = 1280           # Base image size (width and height)
BORDER = 160              # Margin from image borders for target placement

# ==================== Interference Parameters ====================
# 1. FALSE TARGETS: Cropped and rotated real targets (no labels)
FALSE_TARGET_PROB = 0.5   # Probability of adding false targets to an image
FALSE_TARGET_COUNT = 24   # Maximum number of false targets per image

# 2. INTERRUPTIONS: Randomly drawn lines and polygons (no labels)
INTERRUPTION_PROB = 0.25  # Probability of adding drawn interruptions
INTERRUPTION_COUNT = 24   # Maximum number of interruptions per image

# 3. BRIGHTNESS VARIATION: Simulate lighting changes
BRIGHTNESS_RANGE = (0.6, 1.6)  # Brightness factor range (1.0 = original)

# Interruption drawing parameters
LINE_WIDTH_RANGE = (2, 8)       # Line thickness range in pixels
POLYGON_SIDES_RANGE = (2, 16)   # Number of polygon vertices range

# 4. GAUSSIAN BLUR: Simulate motion blur or out-of-focus
BLUR_PROB = 0.3                 # Probability of applying blur
BLUR_RADIUS_RANGE = (0.3, 1.0)  # Gaussian blur radius range

# 5. RESIZE: Downsample some images for scale variation
RESIZE_PROB = 0.3               # Probability of resizing
RESIZE_SIZE = 640               # Target size when resized

# 6. FALSE IMAGES: Paste pre-collected external images (no labels)
FALSE_IMAGE_PROB = 0.5          # Probability of adding false images
FALSE_IMAGE_COUNT = 24          # Maximum number of false images per image

# random seed
random.seed(1337)
# ==================== Directory Setup ====================
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BG_DIR = "./background"                    # Background images directory
TARGET_GENERATED_DIR = "./target" # Real target images directory
OUTPUT_DIR = "./data0"                     # Output directory for generated dataset
FALSE_IMAGE_DIR = "./false_images"         # External false images directory

# ==================== Cache Background Images ====================
# Collect all background image files
bg_files = [f for f in os.listdir(BG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
bg_cache = {}
total_bg = min(1000, len(bg_files))

print(f"Cached images: (0/{total_bg})", end="", flush=True)
for i, f in enumerate(bg_files[:400], 1):
    img = Image.open(os.path.join(BG_DIR, f)).convert('RGBA')
    bg_cache[f] = img.resize((IMG_SIZE, IMG_SIZE))
    print(f"\rCached images: ({i}/{total_bg})", end="", flush=True)
print()

# ==================== Cache Target Images ====================
# Cache real target images separately for train and validation splits
target_cache = {'train': {}, 'val': {}}
print("BACKGROUND CACHED")

for split in ['train', 'val']:
    for cls in range(NC):
        cls_dir = os.path.join(TARGET_GENERATED_DIR, split, str(cls))
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                target_cache[split][cls] = [Image.open(os.path.join(cls_dir, f)).convert('RGBA') for f in files]

# ==================== Create Output Directories ====================
for split in ['train', 'val']:
    for sub in ['images', 'labels']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

# ==================== Helper Functions ====================

def re_size(img):
    """Randomly resize image to lower resolution"""
    if random.random() < RESIZE_PROB:
        return img.resize((RESIZE_SIZE, RESIZE_SIZE), Image.LANCZOS)
    return img

def adjust_brightness(image, brightness_factor):
    """
    Adjust image brightness while preserving alpha channel
    
    Args:
        image: PIL Image (RGBA or RGB)
        brightness_factor: Float multiplier for brightness (1.0 = original)
    
    Returns:
        Brightness-adjusted image
    """
    if brightness_factor == 1.0:
        return image.copy()
    
    # Handle RGBA images separately to preserve transparency
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        rgb = Image.merge('RGB', (r, g, b))
        rgb = Image.blend(rgb, Image.new('RGB', rgb.size, 0), 1 - brightness_factor)
        return Image.merge('RGBA', (*rgb.split(), a))
    else:
        return Image.blend(image, Image.new(image.mode, image.size, 0), 1 - brightness_factor)

def add_image_interference(img):
    """
    Paste random external images as interference (no labels generated)
    
    Args:
        img: Background image to add interference to
    
    Returns:
        Image with pasted false images
    """
    if random.random() > FALSE_IMAGE_PROB or not os.path.exists(FALSE_IMAGE_DIR):
        return img
    
    files = [f for f in os.listdir(FALSE_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return img
    
    for _ in range(random.randint(1, FALSE_IMAGE_COUNT)):
        img_file = random.choice(files)
        target = Image.open(os.path.join(FALSE_IMAGE_DIR, img_file)).convert('RGBA')
        
        # Apply random brightness adjustment
        brightness = random.uniform(*BRIGHTNESS_RANGE)
        target = adjust_brightness(target, brightness)
        
        # Random scaling
        scale = random.uniform(MIN_SIZE / max(target.size), MAX_SIZE / max(target.size))
        w = int(target.width * scale)
        h = int(target.height * scale)
        target = target.resize((w, h))
        
        # Random rotation
        angle = random.uniform(0, 360)
        target = target.rotate(angle, expand=True)
        
        # Random position
        paste_x = random.randint(0, IMG_SIZE - target.width)
        paste_y = random.randint(0, IMG_SIZE - target.height)
        
        # Composite onto image with alpha blending
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        layer.paste(target, (paste_x, paste_y), target)
        img = Image.alpha_composite(img, layer)
    
    return img

def false_target(img, target_cache_split):
    """
    Add cropped and rotated real targets as false positives (no labels)
    
    Process: Rotate -> Crop to 1/3 size -> Rotate again to create fragments
    
    Args:
        img: Background image
        target_cache_split: Cache of real targets for current split
    
    Returns:
        Image with false target fragments
    """
    if random.random() > FALSE_TARGET_PROB or not target_cache_split:
        return img
    
    for _ in range(random.randint(1, FALSE_TARGET_COUNT)):
        # Randomly select a target class
        cls = random.randint(0, NC - 1)
        if cls not in target_cache_split or not target_cache_split[cls]:
            continue
        
        target = random.choice(target_cache_split[cls]).copy()
        
        # Apply brightness variation
        brightness = random.uniform(*BRIGHTNESS_RANGE)
        target = adjust_brightness(target, brightness)
        
        # Random scaling
        scale = random.uniform(MIN_SIZE / max(target.size), MAX_SIZE / max(target.size))
        w = int(target.width * scale)
        h = int(target.height * scale)
        target = target.resize((w, h))
        
        # First rotation
        angle1 = random.randint(1, 360)
        target = target.rotate(angle1, expand=True)
        
        # Crop to 1/3 of original size (simulate partial occlusion)
        cut_ratio = 1/3
        cut_w = int(target.width * cut_ratio)
        cut_h = int(target.height * cut_ratio)
        x1 = random.randint(0, target.width - cut_w)
        y1 = random.randint(0, target.height - cut_h)
        x2 = x1 + cut_w
        y2 = y1 + cut_h
        target = target.crop((x1, y1, x2, y2))
        
        # Second rotation
        angle2 = random.randint(1, 360)
        target = target.rotate(angle2, expand=True)
        
        # Random position
        paste_x = random.randint(0, IMG_SIZE - target.width)
        paste_y = random.randint(0, IMG_SIZE - target.height)
        
        # Composite onto image
        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        layer.paste(target, (paste_x, paste_y), target)
        img = Image.alpha_composite(img, layer)
    
    return img

def draw_interruption(img):
    """
    Draw random geometric shapes (lines and polygons) as visual noise
    
    Args:
        img: Background image
    
    Returns:
        Image with drawn interruptions
    """
    if random.random() > INTERRUPTION_PROB:
        return img
    
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img, 'RGBA')
    
    for _ in range(random.randint(1, INTERRUPTION_COUNT)):
        shape_type = random.choice(['line', 'polygon'])
        
        # Random semi-transparent colors (red, blue, green, black, white)
        color = random.choice([
            (255, random.randint(0, 50), random.randint(0, 50), random.randint(200, 255)),    # Red
            (random.randint(0, 50), random.randint(0, 50), 255, random.randint(200, 255)),    # Blue
            (random.randint(0, 50), 255, random.randint(0, 50), random.randint(200, 255)),    # Green
            (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50), random.randint(200, 255)),  # Black
            (255, 255, 255, random.randint(200, 255))  # White
        ])
        
        if shape_type == 'line':
            # Generate random line segment
            x1 = random.randint(0, IMG_SIZE)
            y1 = random.randint(0, IMG_SIZE)
            x2 = random.randint(0, IMG_SIZE)
            y2 = random.randint(0, IMG_SIZE)
            width = random.randint(*LINE_WIDTH_RANGE)
            
            # Add random jitter to simulate hand-drawn lines (30% probability)
            if random.random() > 0.7:
                points = []
                length = 5 * math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > 0:
                    num_points = max(3, int(length / 50))
                    for i in range(num_points):
                        t = i / (num_points - 1) if num_points > 1 else 0
                        x = x1 + (x2 - x1) * t + random.randint(-10, 10)
                        y = y1 + (y2 - y1) * t + random.randint(-10, 10)
                        points.append((x, y))
                    
                    if len(points) >= 2:
                        draw.line(points, fill=color, width=width)
            else:
                # Straight line
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        
        elif shape_type == 'polygon':
            # Generate random polygon
            num_sides = random.randint(*POLYGON_SIDES_RANGE)
            points = []
            
            # Generate points within a confined area
            center_x = random.randint(100, IMG_SIZE - 100)
            center_y = random.randint(100, IMG_SIZE - 100)
            max_radius = random.randint(50, 200)  # Control polygon size
            
            for _ in range(num_sides):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(20, max_radius)
                
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                
                # Clip to image boundaries
                x = max(0, min(IMG_SIZE, x))
                y = max(0, min(IMG_SIZE, y))
                
                points.append((x, y))
            
            # Sort points by angle to reduce self-intersection (70% probability)
            if random.random() > 0.3:
                center_x = sum(p[0] for p in points) / num_sides
                center_y = sum(p[1] for p in points) / num_sides
                sorted_points = sorted(points, key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))
                points = sorted_points
            
            # Decide whether to add border
            if random.random() > 0.3:  # 70% probability with border
                # Generate border color (slightly lighter or darker)
                outline_color = (
                    max(0, min(255, color[0] + random.randint(-60, 60))),
                    max(0, min(255, color[1] + random.randint(-60, 60))),
                    max(0, min(255, color[2] + random.randint(-60, 60))),
                    color[3]  # Keep same transparency
                )
                outline_width = random.randint(2, 6)
                
                # Fill polygon with semi-transparent color
                fill_color = color[:3] + (color[3] // 2,)
                draw.polygon(points, fill=fill_color)
                
                # Draw border
                closed_points = points + [points[0]]
                draw.line(closed_points, fill=outline_color, width=outline_width)
            else:
                # Border only, no fill
                outline_width = random.randint(2, 6)
                closed_points = points + [points[0]]
                draw.line(closed_points, fill=color, width=outline_width)
    
    # Blend drawn shapes with original image
    blend_alpha = random.uniform(0.8, 1.0)
    return Image.blend(img, draw_img, alpha=blend_alpha)

def apply_blur_if_needed(img):
    """
    Apply Gaussian blur with random radius based on probability
    
    Args:
        img: Input image
    
    Returns:
        Blurred image or original if blur not applied
    """
    if random.random() > BLUR_PROB:
        return img
    
    blur_radius = random.uniform(*BLUR_RADIUS_RANGE)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return blurred_img

def generate_dataset(num_images, split):
    """
    Main dataset generation function
    
    Args:
        num_images: Number of images to generate
        split: 'train' or 'val' - determines which target cache to use
    """
    for idx in range(num_images):
        # Load random background
        bg_name = random.choice(list(bg_cache.keys()))
        img = bg_cache[bg_name].copy()
        
        # Apply interference layers
        img = draw_interruption(img)                    # Add drawn shapes
        img = false_target(img, target_cache[split])    # Add cropped real targets
        img = add_image_interference(img)               # Add external false images
        
        # Define four quadrants for target placement
        regions = [
            # Top-left quadrant
            (random.randint(BORDER, IMG_SIZE // 2 - MAX_SIZE), 
             random.randint(BORDER, IMG_SIZE // 2 - MAX_SIZE)),
            # Top-right quadrant
            (random.randint(IMG_SIZE // 2 + BORDER, IMG_SIZE - MAX_SIZE - BORDER), 
             random.randint(BORDER, IMG_SIZE // 2 - MAX_SIZE)),
            # Bottom-left quadrant
            (random.randint(BORDER, IMG_SIZE // 2 - MAX_SIZE), 
             random.randint(IMG_SIZE // 2 + BORDER, IMG_SIZE - MAX_SIZE - BORDER)),
            # Bottom-right quadrant
            (random.randint(IMG_SIZE // 2 + BORDER, IMG_SIZE - MAX_SIZE - BORDER), 
             random.randint(IMG_SIZE // 2 + BORDER, IMG_SIZE - MAX_SIZE - BORDER))
        ]
        
        labels = []
        for (x_center, y_center) in regions:
            # Random class for this target
            cls = random.randint(0, NC - 1)
            if cls in target_cache[split] and target_cache[split][cls]:
                target_orig = random.choice(target_cache[split][cls]).copy()
                
                # Apply brightness adjustment
                brightness = random.uniform(*BRIGHTNESS_RANGE)
                target_orig = adjust_brightness(target_orig, brightness)
                
                # Random scaling
                scale = random.uniform(MIN_SIZE / max(target_orig.size), MAX_SIZE / max(target_orig.size))
                w_orig = int(target_orig.width * scale)
                h_orig = int(target_orig.height * scale)
                target = target_orig.resize((w_orig, h_orig))
                
                # Random rotation
                angle = random.uniform(0, 360)
                target = target.rotate(angle, expand=True)
                
                # Paste target centered at region coordinates
                paste_x = x_center - target.width // 2
                paste_y = y_center - target.height // 2
                
                layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
                layer.paste(target, (paste_x, paste_y), target)
                img = Image.alpha_composite(img, layer)
                
                # Calculate bounding box corners for YOLO format
                # Original corners relative to target center (before rotation)
                corners = np.array([
                    [-w_orig / 2, -h_orig / 2],  # Top-left
                    [w_orig / 2, -h_orig / 2],   # Top-right
                    [w_orig / 2, h_orig / 2],    # Bottom-right
                    [-w_orig / 2, h_orig / 2]    # Bottom-left
                ])
                
                # Apply rotation matrix
                rad = math.radians(angle)
                rot_mat = np.array([
                    [math.cos(rad), math.sin(rad)], 
                    [-math.sin(rad), math.cos(rad)]
                ])
                
                corners_rotated = np.dot(corners, rot_mat.T)
                
                # Translate to image coordinates and normalize to [0, 1]
                corners_rotated[:, 0] += x_center
                corners_rotated[:, 1] += y_center
                corners_flat = corners_rotated.flatten() / IMG_SIZE
                
                # Clip to valid range
                corners_flat = np.clip(corners_flat, 0.0, 1.0)
                
                # Format: class x1 y1 x2 y2 x3 y3 x4 y4 (YOLO polygon format)
                labels.append(f"{cls} " + " ".join(f"{c:.6f}" for c in corners_flat))
        
        # Apply post-processing effects
        img = apply_blur_if_needed(img)
        img = re_size(img)
        
        # Save image and label files
        img_path = os.path.join(OUTPUT_DIR, split, 'images', f'{idx + 1}.jpg')
        img.convert('RGB').save(img_path)
        
        with open(os.path.join(OUTPUT_DIR, split, 'labels', f'{idx + 1}.txt'), 'w') as f:
            f.write('\n'.join(labels))
        
        # Progress indicator
        print(f"\rGenerating {split}: {idx + 1}/{num_images}  ", end="", flush=True)

# ==================== Main Execution ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Generating TRAIN dataset...")
    print(f"Parameters:")
    print(f"  False target probability: {FALSE_TARGET_PROB}")
    print(f"  False target count: {FALSE_TARGET_COUNT}")
    print(f"  Interruption probability: {INTERRUPTION_PROB}")
    print(f"  Interruption count: {INTERRUPTION_COUNT}")
    print(f"  Brightness range: {BRIGHTNESS_RANGE}")
    print(f"  Blur probability: {BLUR_PROB}")
    print(f"  Blur size range: ({BLUR_RADIUS_RANGE})")
    print(f"  Target directories:")
    print(f"    Train: {os.path.join(TARGET_GENERATED_DIR, 'train')}")
    print(f"    Val: {os.path.join(TARGET_GENERATED_DIR, 'val')}")
    print("=" * 50)
    
    generate_dataset(TRAIN_NUM, 'train')
    
    print("\n" + "=" * 50)
    print("Generating VAL dataset...")
    print("=" * 50)
    
    generate_dataset(VAL_NUM, 'val')
    
    print("\n" + "=" * 50)
    print("FINISHED!")
    print(f"Total images generated: {TRAIN_NUM + VAL_NUM}")
    print(f"  Train: {TRAIN_NUM} (using targets from {os.path.join(TARGET_GENERATED_DIR, 'train')})")
    print(f"  Validation: {VAL_NUM} (using targets from {os.path.join(TARGET_GENERATED_DIR, 'val')})")
    print("=" * 50)
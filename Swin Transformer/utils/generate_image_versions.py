import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

image_folder = Path('/Volumes/TOSHIBA/Github Repositories/transformers/Swin Transformer/images/input') # Input images 
print("Image folder:", image_folder.resolve())

def create_directories(image_folder):
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.join(image_folder, 'low_quality'), exist_ok=True)
    os.makedirs(os.path.join(image_folder, 'color_noise'), exist_ok=True)
    os.makedirs(os.path.join(image_folder, 'gray_noise'), exist_ok=True)
    os.makedirs(os.path.join(image_folder, 'color_jpeg'), exist_ok=True)
    os.makedirs(os.path.join(image_folder, 'gray_jpeg'), exist_ok=True)

def downsize_image(image, scale, name, image_folder):
    """Create downsized version of the image"""
    height, width = image.shape[:2]
    new_height = height // scale
    new_width = width // scale

    # Resize using INTER_AREA for downsizing (good quality)
    downsized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


    # Save downsized image
    output_path = f'{image_folder}/low_quality/{name}_{scale}x.png'
    cv2.imwrite(output_path, downsized)
    print(f"Saved: {output_path}")


def add_gaussian_noise(image, noise_level):
    """Add Gaussian noise to image"""
    # Convert to float32 for noise addition
    image_float = image.astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)

    # Add noise and clip values to valid range
    noisy_image = np.clip(image_float + noise, 0, 255).astype(np.uint8)

    return noisy_image

def create_noisy_versions(image, name, image_folder):
    """Create noisy versions with different noise levels"""
    noise_levels = [15, 25, 50]

    # Convert BGR to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for noise_level in noise_levels:
        # Color noisy version
        color_noisy = add_gaussian_noise(image_rgb, noise_level)
        # Convert back to BGR for OpenCV saving
        color_noisy_bgr = cv2.cvtColor(color_noisy, cv2.COLOR_RGB2BGR)
        color_output_path = f'{image_folder}/color_noise/{name}_noise_{noise_level}.png'
        cv2.imwrite(color_output_path, color_noisy_bgr)
        print(f"Saved: {color_output_path}")

        # Grayscale noisy version
        gray_noisy = add_gaussian_noise(gray_image, noise_level)
        gray_output_path = f'{image_folder}/gray_noise/{name}_noise_{noise_level}.png'
        cv2.imwrite(gray_output_path, gray_noisy)
        print(f"Saved: {gray_output_path}")

def create_jpeg_compressed_versions(image, name, image_folder):
    """Create JPEG compressed versions with different quality levels"""
    compression_levels = [10, 20, 30, 40]

    # Convert BGR to RGB for PIL processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Convert to grayscale PIL image
    gray_pil = pil_image.convert('L')

    for quality in compression_levels:
        # Color JPEG compressed version
        color_output_path = f'{image_folder}/color_jpeg/{name}_color_jpeg_{quality}.png'
        pil_image.save(color_output_path, 'JPEG', quality=quality)
        print(f"Saved: {color_output_path}")

        # Grayscale JPEG compressed version
        gray_output_path = f'{image_folder}/gray_jpeg/{name}_gray_jpeg_{quality}.png'
        gray_pil.save(gray_output_path, 'JPEG', quality=quality)
        print(f"Saved: {gray_output_path}")
    
def process_image(image_path, image_folder):
  image = cv2.imread(image_path)
  if image is None:
    print(f"Error: Unable to read image from {image_path}")
    return

  name = os.path.splitext(os.path.basename(image_path))[0]
  print(f"Processing image: {name}")
  print(f"Original image size: {image.shape[1]}x{image.shape[0]}")

  create_directories(image_folder=image_folder)

  # 1. Create downsized versions (2x, 3x, 4x, 8x)
  print("\n=== Creating downsized versions ===")
  downsize_factors = [2, 3, 4, 8]
  for factor in downsize_factors:
      downsize_image(image, factor, name, image_folder)


  # 2. Create noisy versions (color and grayscale)
  print("\n=== Creating noisy versions ===")
  create_noisy_versions(image, name, image_folder)

  # 3. Create JPEG compressed versions
  print("\n=== Creating JPEG compressed versions ===")
  create_jpeg_compressed_versions(image, name, image_folder)

  print(f"\nProcessing complete! All versions saved for {name}")

image_original_dir = '/Volumes/TOSHIBA/Github Repositories/transformers/Swin Transformer/images/original'

for filename in os.listdir(image_original_dir):
    if filename.endswith('.jpeg'):
        img_path = os.path.join(image_original_dir, filename)
        try:
            #img = Image.open(img_path)
            print(f"Processing image: {filename}")
            process_image(img_path, image_folder)
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

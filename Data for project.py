import kagglehub
import os
import random
from PIL import Image, ImageChops
import re
import numpy as np

# Download latest version (Set this to False the first time you are running the script to download the data)
is_downloaded = True

if is_downloaded:
    print("Data is already downloaded")
else:
    path = kagglehub.dataset_download("pinstripezebra/google-streetview-top-50-us-cities")
    print("Path to dataset files:", path)

# Set the directory containing your images (Change this to your directory)
image_folder = r"C:\Users\nickl\.cache\kagglehub\datasets\pinstripezebra\google-streetview-top-50-us-cities\versions\1\Images"
output_folder = r"C:\Users\nickl\.cache\kagglehub\datasets\processed_images"

os.makedirs(output_folder, exist_ok=True)

target_size = (227, 227)
max_images = 5000  # Stop processing after this many images are created
counter = 0  # Initialize a counter to track the number of saved images

# Load the reference "no imagery" image (Change this to your directory)
reference_image_path = r"C:\Users\nickl\.cache\kagglehub\datasets\pinstripezebra\google-streetview-top-50-us-cities\versions\1\No_photo.jpg"
reference_image = Image.open(reference_image_path)

def is_no_imagery_image(img_path, reference_image):
    try:
        img = Image.open(img_path)
        return ImageChops.difference(reference_image, img).getbbox() is None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

# Dictionary to keep track of the number of files per city
city_counts = {}

# Loop over each file in the directory
for filename in os.listdir(image_folder):
    if counter >= max_images:
        break

    file_path = os.path.join(image_folder, filename)
    
    # Skip files that match the "no imagery" reference image
    if is_no_imagery_image(file_path, reference_image):
        print(f"Deleting: {file_path}")
        os.remove(file_path)
        continue
    
    # Match the pattern "City_ (lat,long).jpg" and extract only the city name
    match = re.match(r"([A-Za-z]+)_ \([-\d.]+, [-\d.]+\)\.jpg", filename)
    if match:
        city_name = match.group(1)

        # Update count for this city
        city_counts[city_name] = city_counts.get(city_name, 0) + 1
        count = city_counts[city_name]

        # Form new filename with unique count if there are multiple images for the same city
        new_filename = f"{city_name}_{count}.jpg" if count > 1 else f"{city_name}.jpg"
        
        old_path = os.path.join(image_folder, filename)
        new_path = os.path.join(image_folder, new_filename)
        
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")

print("Cleanup complete.")

# Function to resize an image to 227x227
def resize_image(img):
    return img.resize(target_size)

# Function to perform a random crop of 227x227 from a 500x500 image
def random_crop(img):
    width, height = img.size
    if width < 227 or height < 227:
        raise ValueError("Image size must be at least 227x227 for cropping.")
    
    x = random.randint(0, width - 227)
    y = random.randint(0, height - 227)
    
    return img.crop((x, y, x + 227, y + 227))

# Function to extract multiple random 227x227 patches from an image
def extract_patches(img, num_patches=5):
    patches = []
    for _ in range(num_patches):
        patch = random_crop(img)
        patches.append(patch)
    return patches

# Loop through each file in the image folder
for filename in os.listdir(image_folder):
    if counter >= max_images:
        break  # Stop if we've reached the max number of images

    file_path = os.path.join(image_folder, filename)
    
    # Open the image
    with Image.open(file_path) as img:
        # Resize the image to 227x227 and save it
        resized_img = resize_image(img)
        resized_img.save(os.path.join(output_folder, f"resized_{filename}"))
        counter += 1
        if counter >= max_images:
            break
        
        # Extract multiple random patches and save them
        patches = extract_patches(img, num_patches=5)
        for i, patch in enumerate(patches):
            if counter >= max_images:
                break
            patch.save(os.path.join(output_folder, f"patch_{i+1}_{filename}"))
            counter += 1
            print(counter)

print("Processing complete. Images and patches saved to output folder.")

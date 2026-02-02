import tifffile
import cv2
import os

# Paths
images_path = "data/data1/D1-1M-sample1-hydrogel.tif"
masks_path = "data/data1/D1-1M-sample1-spheroid.tif"
output_folder = "images"
index = 200  # Index of the image/mask to extract

# Extract the parent folder name (data1 → i=1)
data_i = os.path.basename(os.path.dirname(images_path))  # e.g.: 'data1'

# Prefix for the filename
prefix = f"{data_i}_"

# Output files
image_filename = f"{prefix}image_{index}.png"
mask_filename = f"{prefix}mask_{index}.png"

# Load the stacks
large_images = tifffile.imread(images_path)
large_masks = tifffile.imread(masks_path)

# Extract the image and mask at the given index
image_i = large_images[index]
mask_i = large_masks[index]

# Create the output folder if necessary
os.makedirs(output_folder, exist_ok=True)

# Normalization function to uint8
def normalize_to_uint8(img):
    if img.dtype == 'uint8':
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# Normalize and save
image_i = normalize_to_uint8(image_i)
mask_i = normalize_to_uint8(mask_i)

cv2.imwrite(os.path.join(output_folder, image_filename), image_i)
cv2.imwrite(os.path.join(output_folder, mask_filename), mask_i)

print(f"[✔] Image saved in {output_folder}/{image_filename}")
print(f"[✔] Mask saved in {output_folder}/{mask_filename}")

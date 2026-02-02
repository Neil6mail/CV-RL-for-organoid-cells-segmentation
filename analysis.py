import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images (adjust paths)
image_path = "images/data1_image_300.png" 
mask_path = "images/data1_mask_300.png"

# 1. Load the images
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def apply_non_local_means_denoising(img, h=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)

def apply_gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

#cleaned_image = apply_gaussian_blur(cleaned_image, ksize=55)
image = apply_non_local_means_denoising(image, h=192, templateWindowSize=7, searchWindowSize=21)

# Verify that the dimensions are identical
assert image.shape == mask.shape, "The images must have the same dimensions"

# 2. Create the organoids_only image
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
organoids_only = cv2.bitwise_and(image, image, mask=binary_mask)

# Get the organoid pixels (where the mask is white)
organoid_pixels = image[binary_mask > 0].flatten()

# 3. Create the background image (invert the mask)
inverted_mask = cv2.bitwise_not(binary_mask)
background = cv2.bitwise_and(image, image, mask=inverted_mask)

# 4. Extract pixel values for statistics
# Get the organoid pixels (where the mask is white)
organoid_pixels = image[binary_mask > 0].flatten()
# Get the background pixels (where the mask is black)
background_pixels = image[binary_mask == 0].flatten()

# 5. Create histograms
plt.figure(figsize=(12, 6))

# Histogram for organoids
plt.subplot(1, 2, 1)
plt.hist(organoid_pixels, bins=256, range=(0, 256), color='blue', alpha=0.7)
plt.title('Intensity Histogram - Organoids')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.grid(True)

# Histogram for the background
plt.subplot(1, 2, 2)
plt.hist(background_pixels, bins=256, range=(0, 256), color='red', alpha=0.7)
plt.title('Intensity Histogram - Background')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.grid(True)

plt.tight_layout()
plt.plot(organoids_only)
#plt.show()

# 6. Save the results
cv2.imwrite("organoids_only.tif", organoids_only)
cv2.imwrite("background_only.tif", background)

# Display basic statistics
print(f"Organoid statistics (n={len(organoid_pixels)} pixels):")
print(f"  Min: {np.min(organoid_pixels)}, Max: {np.max(organoid_pixels)}")
print(f"  Mean: {np.mean(organoid_pixels):.2f}, Std deviation: {np.std(organoid_pixels):.2f}")

print(f"\nBackground statistics (n={len(background_pixels)} pixels):")
print(f"  Min: {np.min(background_pixels)}, Max: {np.max(background_pixels)}")
print(f"  Mean: {np.mean(background_pixels):.2f}, Std deviation: {np.std(background_pixels):.2f}")

# Calculate thresholds
# For organoids: value where 80% of pixels are higher = 20th percentile
threshold_organoid = np.percentile(organoid_pixels, 5)
# For the background: value where 20% of pixels are higher = 80th percentile
threshold_background = np.percentile(background_pixels, 95)

print(f"Organoid threshold (95% of pixels >): {threshold_organoid:.1f}")
print(f"Background threshold (5% of pixels >): {threshold_background:.1f}")

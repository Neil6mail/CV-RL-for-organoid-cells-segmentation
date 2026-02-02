import cv2
import numpy as np
from skimage import exposure, filters, morphology, restoration
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, gabor

print("initializing computer vision algo...")


# 1. Base functions
def apply_nothing(img):
    return img

# 2. Filtering / Denoising
def apply_gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_median_blur(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def apply_bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def apply_non_local_means_denoising(img, h=10, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)

def apply_anisotropic_diffusion(img, niter=10, kappa=50, gamma=0.1):
    return restoration.denoise_tv_chambolle(img, weight=gamma, max_num_iter=niter)

# 3. Histogram & Contrast
def apply_clahe(img, clip_limit=2.0, tile_grid_size=8):

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(img)

def apply_histogram_equalization(img):
    return cv2.equalizeHist(img)

def apply_contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

def apply_gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    return (c * np.log(1 + img)).astype(np.uint8)

def apply_power_law_transform(img, gamma=0.5):
    img = img / 255.0
    img = np.power(img, gamma)
    return (img * 255).astype(np.uint8)

# 4. Edge Detection
def apply_sobel_edge(img, dx=1, dy=0, ksize=3):
    if dx == 1 and dy == 1:
        print("Applying Sobel filter in both directions")
        sobelx = cv2.Sobel(img, cv2.CV_64F,1, 0, ksize=ksize)  # Gradient horizontal
        sobely = cv2.Sobel(img, cv2.CV_64F,0, 1, ksize=ksize)  # Gradient vertical
        result=cv2.magnitude(sobelx, sobely)
        
        return result.astype(np.uint8)  # Combine les deux
    
    print("Applying Sobel filter in 1 directions")
    return cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize).astype(np.uint8)

def apply_canny(img, threshold1=100, threshold2=200):
    return cv2.Canny(img, threshold1, threshold2)

def apply_laplacian(img, ksize=3):
    return cv2.Laplacian(img, cv2.CV_64F, ksize=ksize).astype(np.uint8)

def apply_unsharp_mask(img, strength=1.5, blur_ksize=5):
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

def apply_difference_of_gaussians(img, low_sigma=1.0, high_sigma=2.0):
    low = cv2.GaussianBlur(img, (0, 0), low_sigma)
    high = cv2.GaussianBlur(img, (0, 0), high_sigma)
    return cv2.subtract(low, high)

def apply_gabor_filter(img, theta=0, sigma=4.0, lambda_=10.0):
    filt_real, _ = gabor(img, frequency=1.0 / lambda_, theta=theta, sigma_x=sigma, sigma_y=sigma)
    return (filt_real * 255).astype(np.uint8)

# 5. Thresholding

def thresholding_range(img, threshold_min=128, threshold_max=255):
    return cv2.inRange(img, threshold_min, threshold_max)

def threshold_filter(img, threshold_min=0, threshold_max=255):
    mask = (img >= threshold_min) & (img <= threshold_max)
    return img * mask  # Multiplie par 1 ou 0


def apply_tresholding(img, threshold_min=128, threshold_max=255):
    return cv2.threshold(img, threshold_min, threshold_max, cv2.THRESH_BINARY)[1]

def apply_adaptive_threshold(img, blockSize=11, C=2):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)

def apply_otsu_threshold(img):
    tresh,image=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("yes:", tresh)
    return image

def apply_threshold_li(img):
    threshold = threshold_li(img)
    return (img > threshold).astype(np.uint8) * 255

def apply_threshold_yen(img):
    threshold = threshold_yen(img)
    return (img > threshold).astype(np.uint8) * 255

# 6. Morphological Ops
def apply_morph_opening(img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def apply_morph_closing(img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def apply_morph_gradient(img, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def apply_remove_small_objects(img, min_size=64):
    return morphology.remove_small_objects(img > 0, min_size=min_size).astype(np.uint8) * 255

def apply_remove_small_holes(img, area_threshold=64):
    return morphology.remove_small_holes(img > 0, area_threshold=area_threshold).astype(np.uint8) * 255

# 7. Top-hat / Black-hat
def apply_top_hat_transform(img, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def apply_black_hat_transform(img, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# 8. Other
def apply_equalize_adapthist(img, clip_limit=0.03):
    return (exposure.equalize_adapthist(img / 255.0, clip_limit=clip_limit) * 255).astype(np.uint8)

def apply_z_score_normalization(img):

    img_cleaned=img[img > 0]
    if len(img_cleaned) == 0:
        return img  # Si l'image est vide, retourne l'image d'origine

    img_mean = np.mean(img_cleaned)
    img_std = np.std(img_cleaned)
    print(img_mean, img_std)
    if img_std == 0:
        return img
    return (((img - img_mean) / img_std) * 64 + 128).clip(0, 255).astype(np.uint8)

def apply_bitwise(img1, img2,operation="AND"):
    if operation == "AND":
        return cv2.bitwise_and(img1, img2)
    elif operation == "OR":
        return cv2.bitwise_or(img1, img2)
    elif operation == "XOR":
        return cv2.bitwise_xor(img1, img2)
    elif operation == "MASK":
        return cv2.bitwise_and(img1, img1, mask=img2)
    elif operation == "NOT":
        return cv2.bitwise_not(img1)
    else:
        raise ValueError("Unknown operation")
    
def apply_contours(image):
    contours= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    height, width = image.shape[:2]
    all_black = np.zeros((height,width), dtype=np.uint8)
    return cv2.drawContours(all_black, contours, -1, 255, 2)


def apply_elliptic_test(image, aspect_ratio_max=2.0,circularity_min=0.6,solidity_min=0.8,noise=100):
    
    """

    Returns:
        numpy.ndarray: L'image d'entrée avec les contours des formes détectées en rouge.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_shape = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < noise:  # Ignorer les petites régions bruitées
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        hull = cv2.convexHull(contour)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        aspect_ratio_ellipse = None

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse
            aspect_ratio_ellipse = float(MA) / ma if ma > 0 else 0

        # Application des tests
        aspect_ratio_valid = aspect_ratio_ellipse is not None and aspect_ratio_ellipse < aspect_ratio_max
        circularity_valid = circularity >= circularity_min
        solidity_valid = solidity >= solidity_min

        if aspect_ratio_valid and circularity_valid and solidity_valid:
            valid_shape.append(contour)

    # Dessiner les contours des formes valides en rouge
    height, width = image.shape[:2]
    all_black = np.zeros((height,width), dtype=np.uint8)
    image_result=cv2.drawContours(all_black, valid_shape, -1, 255, 2)

    return image_result

def restrictive_filter(image, diameter=3, mediane=80, sdt=10):
    pass



ALGO_REGISTRY_30 = {
    "raw": {
        "func": apply_nothing,
        "params": {}
    },
    "gaussian_blur": {
        "func": apply_gaussian_blur,
        "params": {"ksize": 5}
    },
    "median_blur": {
        "func": apply_median_blur,
        "params": {"ksize": 5}
    },
    "bilateral_filter": {
        "func": apply_bilateral_filter,
        "params": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}
    },
    "clahe": {
        "func": apply_clahe,
        "params": {"clip_limit": 2.0, "tile_grid_size": 8}
    },
    "histogram_equalization": {
        "func": apply_histogram_equalization,
        "params": {}
    },
    "sobel_edge": {
        "func": apply_sobel_edge,
        "params": {"dx": 1, "dy": 0, "ksize": 3}
    },
    "canny": {
        "func": apply_canny,
        "params": {"threshold1": 100, "threshold2": 200}
    },
    "laplacian": {
        "func": apply_laplacian,
        "params": {"ksize": 3}
    },
    "unsharp_mask": {
        "func": apply_unsharp_mask,
        "params": {"strength": 1.5, "blur_ksize": 5}
    },
    "tresholding": {
        "func": apply_tresholding,
        "params": {"threshold_min": 70, "threshold_max": 255}
    },
    "thresholding_range": {
        "func": thresholding_range,
        "params": {"threshold_min": 128, "threshold_max": 255}
    },
    "threshold_filter": {
        "func": threshold_filter,
        "params": {"threshold_min": 0, "threshold_max": 255}
    },
    "adaptive_threshold": {
        "func": apply_adaptive_threshold,
        "params": {"blockSize": 15, "C": 5}
    },

    "otsu_threshold": {
        "func": apply_otsu_threshold,
        "params": {}
    },
    "contrast_stretching": {
        "func": apply_contrast_stretching,
        "params": {}
    },
    "gamma_correction": {
        "func": apply_gamma_correction,
        "params": {"gamma": 1.5}
    },
    "log_transform": {
        "func": apply_log_transform,
        "params": {}
    },
    "power_law_transform": {
        "func": apply_power_law_transform,
        "params": {"gamma": 0.5}
    },
    "gabor_filter": {
        "func": apply_gabor_filter,
        "params": {"theta": 0, "sigma": 4.0, "lambda_": 10.0}
    },
    "difference_of_gaussians": {
        "func": apply_difference_of_gaussians,
        "params": {"low_sigma": 1.0, "high_sigma": 2.0}
    },
    "non_local_means_denoising": {
        "func": apply_non_local_means_denoising,
        "params": {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21}
    },
    "anisotropic_diffusion": {
        "func": apply_anisotropic_diffusion,
        "params": {"niter": 10, "kappa": 50, "gamma": 0.1}
    },
    "threshold_li": {
        "func": apply_threshold_li,
        "params": {}
    },
    "threshold_yen": {
        "func": apply_threshold_yen,
        "params": {}
    },
    "morph_opening": {
        "func": apply_morph_opening,
        "params": {"kernel_size": 3}
    },
    "morph_closing": {
        "func": apply_morph_closing,
        "params": {"kernel_size": 3}
    },
    "morph_gradient": {
        "func": apply_morph_gradient,
        "params": {"kernel_size": 3}
    },
    "remove_small_objects": {
        "func": apply_remove_small_objects,
        "params": {"min_size": 32}
    },
    "remove_small_holes": {
        "func": apply_remove_small_holes,
        "params": {"area_threshold": 64}
    },
    "top_hat_transform": {
        "func": apply_top_hat_transform,
        "params": {"kernel_size": 15}
    },
    "black_hat_transform": {
        "func": apply_black_hat_transform,
        "params": {"kernel_size": 15}
    },
    "equalize_adapthist": {
        "func": apply_equalize_adapthist,
        "params": {"clip_limit": 0.03}
    },
    "z_score_normalization": {
        "func": apply_z_score_normalization,
        "params": {}
    },
    "bitwise": {
        "func": apply_bitwise,
        "params": {"operation": "NOT"}  # Tu peux même switch entre AND/OR/XOR
    },
    "contours": {
        "func": apply_contours,
        "params": {}
    },
    "elliptic_test": {
        "func": apply_elliptic_test,
        "params": {"aspect_ratio_max": 2.0, "circularity_min": 0.6, "solidity_min": 0.8, "noise": 100}
    }
}
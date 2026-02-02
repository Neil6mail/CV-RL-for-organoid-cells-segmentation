import numpy as np
from sklearn.metrics import f1_score

def compute_f1_score(prediction: np.ndarray, mask: np.ndarray, threshold: int = 128) -> float:
    """
    Compute the F1-Score between a predicted image and a ground truth mask.

    Args:
        prediction (np.ndarray): Predicted image (grayscale or binary).
        mask (np.ndarray): Ground truth mask (binary or grayscale).
        threshold (int): Threshold to apply for binarization if necessary.

    Returns:
        float: F1-Score (between 0 and 1).
    """

    # Flatten the arrays
    pred_flat = prediction.flatten()
    mask_flat = mask.flatten()

    # Binarize if necessary
    if pred_flat.max() > 1:
        pred_flat = (pred_flat >= threshold).astype(np.uint8)
    if mask_flat.max() > 1:
        mask_flat = (mask_flat >= threshold).astype(np.uint8)

    # Compute the F1-score
    score = f1_score(mask_flat, pred_flat, zero_division=0)
    return score

def pixel_distribution(image, mask, percentile=0.95,thresh=59): 
    """
    Statistics on organoids and environment.
    """

    organoid_pixels = image[(mask > 0) & (image > 0)]
    background_pixels = image[(mask == 0) & (image > 0)]
    if background_pixels.size == 0:
        background_pixels = [0]

    # Statistics
    stats = {"organoid": {}, "background": {}}
    for key, pixels in zip(["organoid", "background"], [organoid_pixels, background_pixels]):
        
        element = stats[key]
        element["min"] = np.min(pixels)
        element["max"] = np.max(pixels)
        element["mean"] = np.mean(pixels)
        element["median"] = np.median(pixels)
        element["quartile"] = np.percentile(pixels, 25)
        element["max_quartile"] = np.percentile(pixels, 75)
        element["std"] = np.std(pixels)
        element["pixel_count"] = len(pixels)
        element[f"{percentile*100:.0f}"] = np.percentile(pixels, percentile*100)
        element[f"{(1-percentile)*100:.0f}"] = np.percentile(pixels, (1-percentile)*100)
        element[f"thresh"] = np.count_nonzero(pixels >= thresh)
        element[f"thresh%"] = int((np.count_nonzero(pixels >= thresh)/len(pixels)*100))
        
        text = f"{element['pixel_count']} positive pixels in {key}:"
        text += f"\n  Min: {element['min']}, Max: {element['max']}"
        text += f"\n  Quartile: {element['quartile']:.2f}, Max quartile: {element['max_quartile']:.2f}"
        text += f"\n  Mean: {element['mean']:.2f}, Median: {element['median']:.2f}, Std deviation: {element['std']:.2f}"
        text += f"\n  Upper threshold of the smallest {percentile*100:.0f}% pixels: {element[f'{percentile*100:.0f}']}"
        text += f"\n  Lower threshold of the largest {percentile*100:.0f}% pixels: {element[f'{(1-percentile)*100:.0f}']}"
        text += f"\n  Number of pixels below {thresh} : {element[f'thresh%']} % ({element[f'thresh']})"
        stats[key]["text"] = text

    return stats




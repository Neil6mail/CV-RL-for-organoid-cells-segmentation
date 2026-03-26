# Organoid Segmentation: Computer Vision and Image Analysis

<div align="center">

**A comprehensive pipeline for precise organoid cell segmentation combining classical image processing with automatic evaluation**

</div>

---
<p align="center">
    <img width="596" height="320" alt="Capture d&#39;écran 2026-03-26 123441" src="https://github.com/user-attachments/assets/24c523d0-de9d-431d-9ebd-4f8e47619953" />
</p>

## Overview

This project provides a **comprehensive research platform** for developing and optimizing organoid segmentation algorithms. It seamlessly integrates:

- **30+ Computer Vision Algorithms** – Filters, edge detection, thresholding methods, and morphological operations
- **Interactive Visualization Interface** – Real-time algorithm exploration and comparison
- **Dual Evaluation System** – Human scoring + automatic MedSAM evaluation with F1-score metrics
- **Production-Ready Pipeline** – From raw TIFF images to annotated segmentation results

---
- **Detailed Report**: See `Report.pdf`
- **Presentation**: See `Presentation slides.pdf`
---

## Key Features

| Feature | Description |
|---------|-------------|
| **Computer Vision Toolkit** | 30+ algorithms including Gaussian blur, Sobel edge detection, Canny, Otsu/Yen/Li thresholding, CLAHE, morphological operations |
| **Interactive Viewer** | Visualize algorithm effects in real-time with human evaluation interface |
| **Dual Scoring System** | F1-score computation + MedSAM automatic segmentation evaluation |
| **Statistical Analysis** | Pixel distribution analysis, organoid/background separation statistics |
| **TIFF Stack Processing** | Extract and normalize images from large microscopy TIFF stacks |
| **Results Management** | Organize results by method with applied algorithms, statistics, and segmented images |

---

## Key Research Insights

Detailed findings are documented in:
- **Report.pdf** – Complete technical analysis and results
- **Presentation slides.pdf** – Visual summary of methods and findings
- **presentation/** folder – Method-specific results and statistics

Notable tested methods:
- Yen's threshold method (strong performance)
- Background cleaning approach (5% organoid loss)
- Sobel edge detection methods
- Custom algorithm sequence optimization

---

## Project Structure

```
CV-RL-for-organoid-cells-segmentation/
│
├── Core Modules
│   ├── computer_vision_algo.py    # 30+ vision algorithms (filters, thresholds, edge detection)
│   ├── features.py                # Main API for applying/evaluating algorithms
│   ├── metric.py                  # F1-score and pixel distribution analysis
│   ├── image_treatment.py         # Image loading, patchification, preprocessing
│   ├── clean_borders.py           # Mask cleaning and border processing
│   └── analysis.py                # Statistical analysis on segmented images
│
├── Entry Points
│   ├── main.py                    # Main script for exploration and manual evaluation
│   ├── show_computer_vision_algo.py # Interactive visualization interface
│   └── export_slice_image.py      # Extract images/masks from TIFF stacks
│
├── ML Components
│   ├── load_medsam.py             # MedSAM model loading and inference
│   └── parameters.py              # Global configuration (paths, patch size, model settings)
│
├── Data & Results
│   ├── images/                    # Input images and ground truth masks
│   ├── evaluation/                # Evaluation scores and analysis figures
│   │   ├── human_scores.csv       # Manual evaluation results
│   │   └── medsam_scores.csv      # Automatic evaluation results
│   ├── presentation/              # Organized results by method
│   │   ├── BACKGROUND CLEANED (5% organoids loss)/
│   │   ├── Elliptic test/
│   │   ├── The Z method/
│   │   ├── Working with sobel edge method/
│   │   ├── Yen's threshold method at its peak/
│   │   └── [Other method folders]/
│   └── __pycache__/               # Python cache
│
├── Setup & Configuration
│   └── to setup the environment/
│       ├── medsam_environment.yaml # Conda environment (recommended)
│       └── requirements.txt        # pip dependencies
│
└── Documentation
    ├── README.md                  # This file
    ├── LICENSE                    # License information
    ├── Report.pdf                 # Detailed project report
    ├── Presentation slides.pdf    # Research presentation
    └── human_scores.csv & medsam_scores.csv # Top-level evaluation results
```

---

## Quick Start

### Prerequisites
- **Python 3.8+**
- **CUDA 11.0+** (for GPU-accelerated MedSAM, optional but recommended)
- **Git**

### Installation

#### Option 1: Conda (Recommended for MedSAM)

```bash
# Clone the repository
git clone <repo-url>
cd CV-RL-for-organoid-cells-segmentation

# Create and activate environment
conda env create -f "to setup the environment/medsam_environment.yaml"
conda activate medsam

# Done! All dependencies are installed
```

#### Option 2: pip (Lightweight)

```bash
# Clone and navigate
git clone <repo-url>
cd CV-RL-for-organoid-cells-segmentation

# Install dependencies
pip install -r "to setup the environment/requirements.txt"

# Note: MedSAM evaluation won't be available without additional setup
```

---

## Usage Guide

### 1. Interactive Algorithm Exploration

Launch the interactive interface to explore algorithms in real-time:

```bash
python main.py
```

**What you can do:**
- Load your image and ground truth mask
- Apply different algorithm sequences interactively
- View results side-by-side
- Perform manual human evaluation
- Export scores to CSV

---

### 2. Batch Evaluation with Scoring

Evaluate algorithm sequences and get automatic MedSAM scores:

```python
from features import eval_all_algo_individualy
from parameters import *

# Define algorithm sequence
algo_sequence = [
    "gaussian_blur", 
    "unsharp_mask", 
    "adaptive_threshold", 
    "otsu_threshold", 
    "canny"
]

# Evaluate with human interface and MedSAM scoring
eval_all_algo_individualy(
    algo_sequence, 
    image_path, 
    mask_path,
    human_eval=True
)
```

---

### 3. TIFF Stack Processing

Extract individual images/masks from large microscopy TIFF stacks:

```bash
python export_slice_image.py
```

This will:
- Read TIFF stacks from `images/` folder
- Extract and normalize slices
- Convert to PNG/grayscale for processing
- Save organized output structure

---

### 4. Mask Cleaning and Analysis

Clean borders and analyze segmentation results:

```python
from clean_borders import clean_mask_borders, remove_small_objects
from analysis import analyze_segmentation

# Clean a mask
cleaned_mask = clean_mask_borders(mask)

# Analyze pixel distributions
stats = analyze_segmentation(image, cleaned_mask)
print(f"Organoid mean intensity: {stats['organoid']['mean']}")
print(f"Background mean intensity: {stats['background']['mean']}")
```

---

## Available Algorithms

### Denoising and Filtering (8)
- `gaussian_blur` – Gaussian smoothing
- `median_blur` – Median filtering
- `bilateral_filter` – Edge-preserving smoothing
- `non_local_means_denoising` – Advanced denoising
- `anisotropic_diffusion` – Structure-preserving diffusion

### Contrast Enhancement (5)
- `clahe` – Adaptive histogram equalization
- `histogram_equalization` – Global intensity normalization
- `contrast_stretching` – Percentile-based stretching
- `gamma_correction` – Non-linear intensity adjustment
- `log_transform` – Logarithmic intensity mapping

### Edge Detection (4)
- `sobel_edge` – Gradient-based edge detection
- `canny` – Multi-stage edge detection
- `laplacian` – Second derivative edge detection
- `unsharp_mask` – High-pass sharpening

### Thresholding (4)
- `otsu_threshold` – Optimal global threshold
- `yen_threshold` – Entropy-based threshold
- `li_threshold` – Ratio-maximization threshold
- `adaptive_threshold` – Local adaptive threshold

### Morphological Operations (4+)
- `erosion` – Shrink foreground regions
- `dilation` – Expand foreground regions
- `opening` – Remove small noise
- `closing` – Fill small holes

**And more!** See `computer_vision_algo.py` for the complete list.

---

## Evaluation Metrics

### F1-Score
Harmonic mean of precision and recall:
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Computed at pixel level against ground truth mask.

### MedSAM Evaluation
Automatic segmentation using the Medical Segment Anything Model:
- Provides independent segmentation predictions
- Useful for assessing segmentation quality
- Can identify algorithm failures

### Pixel Distribution Analysis
Statistical analysis of organoid vs. background pixels:
- Min/max/mean/median intensities
- Quartile information
- Overlap statistics for evaluating separation

---

## Results Structure

Results are organized hierarchically for easy comparison:

```
presentation/
├── BACKGROUND CLEANED (5% organoids loss)/
│   ├── segmented_images/
│   ├── applied_algorithms.csv      # Sequence of algorithms used
│   └── stats.csv                   # Evaluation metrics
│
├── Yen's threshold method at its peak/
│   ├── segmented_images/
│   ├── applied_algorithms.csv
│   └── stats.csv
│
└── [Other methods...]/
```

Each method folder contains:
- **Segmented images** – Visual results
- **applied_algorithms.csv** – Algorithm sequences with parameters
- **stats.csv** – Performance metrics (F1-score, MedSAM scores, etc.)

---



## Scientific Background

This project targets **organoid segmentation** in biomedical imaging:

- **Organoids**: Self-organizing 3D cellular structures derived from stem cells
- **Challenge**: Precise segmentation of organoid boundaries in microscopy images
- **Approach**: Combine classical computer vision (fast, interpretable) with modern ML (accurate)
- **Goal**: Develop robust, generalizable segmentation pipelines

The dual-evaluation system (human + MedSAM) ensures reliability for research use.

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{organoid_segmentation_2024,
  title = {Organoid Segmentation: Computer Vision + Reinforcement Learning},
  author = {neil6mail},
  year = {2024},
  url = {<repo-url>}
}
```

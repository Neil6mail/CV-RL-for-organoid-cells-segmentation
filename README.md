# Organoid Segmentation with Computer Vision and Reinforcement Learning

This project provides a complete platform for exploring, optimizing, and evaluating organoid segmentation methods, combining classic computer vision algorithms and custom reinforcement learning (RL) agents. It includes an interactive interface, analysis scripts, human and automatic evaluation tools (MedSAM), and a notebook for RL experimentation.

## Goals

- Explore and compare image processing algorithms for organoid segmentation.
- Optimize algorithm sequences and their parameters via custom RL agents.
- Evaluate segmentation quality with both human and automatic scores (F1-score, MedSAM).
- Provide an extensible base for research in biomedical image segmentation.

---

## Repository Structure

```
├── analysis.py                  # Statistical analysis on segmented images
├── clean_borders.py             # Border and mask cleaning
├── computer_vision_algo.py      # Vision algorithm implementations (filters, thresholds, etc.)
├── export_slice_image.py        # Extraction/normalization of images/masks from TIFF stacks
├── features.py                  # Main functions for applying/evaluating algorithms
├── human_scores.csv             # Exported human evaluation scores
├── image_treatment.py           # Loading, patchification, preprocessing
├── load_medsam.py               # MedSAM model loading and usage
├── main.py                      # Main script for exploration and evaluation
├── metric.py                    # Scoring functions (F1, pixel distribution, etc.)
├── parameters.py                # Global parameters (paths, patch size, MedSAM activation...)
├── RL.ipynb                     # RL experimentation notebook
├── show_computer_vision_algo.py # Interactive interface to explore algorithms
├── images/                      # Input images and masks
├── evaluation/                  # Evaluation results (scores, images, stats)
├── presentation/                # Results folders, figures, CSVs, stats by method
├── to setup the environment/    # Environment files (requirements.txt, medsam_environment.yaml)
└── ...
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd CV-RL-for-organoid-cells-segmentation
```

### 2. Install dependencies

#### a) With Conda (recommended for MedSAM)

```bash
conda env create -f "to setup the environment/medsam_environment.yaml"
conda activate medsam
```

#### b) With pip (for use without MedSAM)

```bash
pip install -r "to setup the environment/requirements.txt"
```

---

## Usage

### 1. Interactive Algorithm Exploration

Run the main script to explore and compare vision algorithms:

```bash
python main.py
```
The interface allows you to visualize results, perform manual evaluation, and export scores.

### 2. RL Notebook

The `RL.ipynb` notebook lets you experiment with optimizing algorithm sequences via RL. Open it in Jupyter or VS Code to test and modify agents.

### 3. Image Extraction and Preparation

Use `export_slice_image.py` to extract images/masks from large TIFF files.

### 4. Cleaning and Analysis

Scripts like `clean_borders.py` and `analysis.py` help refine masks and analyze pixel distributions.

---

## Data and Results Structure

- **images/**: Input images and binary masks.
- **presentation/**: Folders by method/test, containing segmented images, applied algorithm CSVs, statistics.
- **evaluation/**: Human and MedSAM scores, analysis figures.
- **human_scores.csv / medsam_scores.csv**: Exported global scores.

---

## Evaluation

- **Human evaluation**: Interactive interface for visually scoring each result, exportable as CSV.
- **Automatic evaluation**: F1-score computed between predicted masks and ground truth, via `metric.py`.
- **MedSAM**: Can be activated for automatic evaluation with a SOTA model (see `parameters.py`).

---

## Customization and Extension

- Add your own algorithms in `computer_vision_algo.py` and reference them in `parameters.py`.
- Modify the algorithm sequences to test in `main.py` or via the interface.
- To train/evaluate MedSAM on your data, adjust paths and parameters in `parameters.py`.

---

## Authors and License

Open-source project under the MIT license. Feel free to contribute or open issues!

---

## Acknowledgments

- Built on open-source tools (OpenCV, scikit-image, PyTorch, transformers, MedSAM...)
- Thanks to the scientific community for datasets and benchmarks.

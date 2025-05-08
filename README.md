<p align="center">
  <img src="thumbnail.png" alt="Visual Localization Thumbnail" width="600"/>
</p>

# Visual Localization using pre-existing Satellite Images

This repository provides a framework for benchmarking different feature matching algorithms (LightGlue, SuperGlue, GIM variants) for the task of visual localization, specifically comparing drone imagery against satellite maps. It includes pipelines for each matcher, preprocessing capabilities (resizing, warping), and calculates localization error in meters. 


## Features

*   **Multiple Matchers:** Compare LightGlue, SuperGlue, and GIM (DKM, RoMa, LoFTR, LightGlue variant).
*   **Standardized Pipelines:** Encapsulated pipelines for easy execution (`src/`).
*   **Preprocessing:** Optional image resizing and perspective warping (top-down view simulation).
*   **Meter-Level Error:** Calculates localization error using Haversine distance between Ground Truth and Prediction.
*   **Detailed Output:** Generates per-pair results (`.txt`), visualizations (`.png`), and overall summaries (`.csv`, `.txt`).

## Getting Started

### 1. Prerequisites

*   **Conda:** Anaconda or Miniconda is required for environment management.

### 2. Clone Repository

Clone this repository *recursively* to include the necessary matcher submodules:

```bash
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
cd SatelliteLocalization
```

If you cloned without --recursive, run this inside the repository directory:
```bash
git submodule update --init --recursive
```
### 3. Setup Environment

Follow the detailed steps in INSTALL.md to create the Conda environment.

### 4. Data Setup

- Place your drone query images in the data/query/ directory.
- Place the corresponding photo_metadata.csv file (containing columns like Filename, Latitude, Longitude, and orientation angles if using warping) in data/query/.
- Place your satellite map tile images in the data/map/ directory.
- Place the corresponding map.csv file (containing columns like Filename, Top_left_lat, etc.) in data/map/.
(Refer to the provided example CSV files for the required format.)

### 5. Model Weights

This repository does not include pre-trained model weights (.ckpt, .pth).

Download the necessary weights for the specific matcher(s) you intend to use (especially for GIM variants).

Place the weights in an accessible location (e.g., within the respective matchers/<matcher_name>/weights directory. However, this might be ignored by the submodule's gitignore - a central weights/ folder outside matchers might be better).

Update the paths in config.yaml under the matcher_weights section accordingly (e.g., gim_weights_path).

### 6. Run Benchmark

Activate your Conda environment (e.g., conda activate matcher_benchmark) and run:
```bash
python benchmark.py --config config.yaml
```
### 7. Output
Results will be saved in a timestamped subdirectory inside data/output/. This includes:
Folders for each query image containing:
Per-map comparison .txt files with detailed metrics.
Match visualization .png files (if enabled and successful).
benchmark_summary.csv: Summary of the best match results for each query.
benchmark_stats.txt: Overall statistics for the benchmark run.
processed_queries/ (if preprocessing is used): Contains the modified query images used for matching.


### Acknowledgments
This framework builds upon concepts demonstrated in WildNav and its implementation.
Utilizes the excellent open-source work from the LightGlue, SuperGlue, and GIM authors.

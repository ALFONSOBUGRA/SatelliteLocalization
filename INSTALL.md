# Installation Guide

This guide details how to set up the Conda environment required to run the Visual Localization Benchmark.

## 1. Prerequisites

Before starting, ensure you have the following installed:

*   **Git:** For cloning the repository and managing submodules. ([Download Git](https://git-scm.com/downloads))
*   **Conda:** Anaconda or Miniconda for environment and package management. ([Download Miniconda](https://docs.conda.io/en/latest/miniconda.html))
*   **(Optional) NVIDIA GPU & Drivers:** If you plan to use GPU acceleration (`device: 'cuda'` in `config.yaml`), you need:
    *   An NVIDIA GPU compatible with the CUDA version specified in `environment.yml` (e.g., CUDA 11.3).
    *   Appropriate NVIDIA drivers installed for your OS and GPU. Verify installation by running `nvidia-smi` in your terminal.

## 2. Clone the Repository

Open your terminal or command prompt and clone the repository **recursively** to ensure the submodules (LightGlue, SuperGlue, GIM) are also downloaded:

```bash
git clone --recursive https://github.com/ALFONSOBUGRA/SatelliteLocalization.git
```
cd SatelliteLocalization

If you already cloned without --recursive: Navigate into the cloned directory and run:
```bash
git submodule update --init --recursive
```

## 3. Create Conda Environment

The required packages and their versions are defined in the environment.yml file.
Navigate to the Repository Root: Make sure your terminal is in the main project directory (e.g., VisualLocalization).
Create Environment: Run the following command. This might take several minutes as Conda resolves dependencies and downloads packages.
conda env create -f environment.yml


This will create a new Conda environment named matcher_benchmark (or the name specified in environment.yml).

Activate Environment: Before running the benchmark script, you must activate the newly created environment:
```bash
conda activate visloc
```
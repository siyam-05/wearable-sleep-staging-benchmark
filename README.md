
# Wearable Sleep Staging Benchmark: Random Forest vs Deep Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

This repository provides the complete code and data pipeline for the paper:

**"Waking Up to the Truth: A Simple Random Forest Outperforms Deep Learning for Wearable Sleep Staging"**  
*Md. Tanvir Hasan Siyam*  
(Submitted to *Biomedical Signal Processing and Control*)

## Overview

We benchmark sleep stage classification (Wake, N1, N2, N3, REM) using only heart rate (HR) and accelerometry (ACC) from 47 subjects. The Random Forest achieves **81% accuracy** and **Cohen’s kappa = 0.64**, outperforming BiLSTM‑attention, SVM, and logistic regression.

## Requirements

- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`

## Data

The dataset is publicly available:  
[Multi‑night instantaneous heart rate and accelerometry dataset with EEG sleep stage labels (v1.0.0)](https://physionet.org/...).  
The script will automatically download or you must place the files in `data/` as described.

## Usage

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/wearable-sleep-staging-benchmark.git
   cd wearable-sleep-staging-benchmark
Install dependencies:

bash
pip install -r requirements.txt
Run the full pipeline:

bash
python src/run_pipeline.py
This will:

Load and preprocess the data

Extract 18 features

Train Random Forest, SVM, LR, BiLSTM

Perform LOSO cross‑validation

Generate all figures (saved in results/figures/)

Save tables (CSV) in results/tables/

Results Summary
Model	Accuracy	Kappa
Random Forest	0.810	0.64
SVM (RBF)	0.431	0.21
Logistic Regression	0.267	0.12
BiLSTM‑Attention	0.370	0.188
All figures and tables from the paper are reproduced.

Citation
If you use this code, please cite


License
MIT

markdown
# Wearable Sleep Staging Benchmark

**Paper:** *Waking Up to the Truth: When Simplicity Prevails — A Validated Random Forest Baseline Challenges Deep Learning for Wearable Five-Class Sleep Staging*  
**Author:** Md. Tanvir Hasan Siyam

This repository provides the complete, reproducible code for the above paper.  
We benchmark five‑class sleep staging (Wake, N1, N2, N3, REM) using only wrist‑worn heart rate and accelerometry from 47 subjects (253 nights). A Random Forest achieves **81.0% accuracy, Cohen’s κ = 0.64**, outperforming BiLSTM‑Attention, SVM, and logistic regression under strict **Leave‑One‑Subject‑Out** cross‑validation.

## Dataset

The study uses the public dataset:  
*Multi‑night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels* (version 1.0.0)  
PhysioNet DOI: [10.13026/a0sy-7t69](https://doi.org/10.13026/a0sy-7t69)

Download and place the contents under `data/` so that each subject folder (e.g., `sub-01/`) contains `hr.csv`, `motion.csv`, and `labels.mat`.

## Installation

```bash
git clone https://github.com/siyam-05/wearable-sleep-staging-benchmark.git
cd wearable-sleep-staging-benchmark
pip install -r requirements.txt
Usage
Set the data path in src/run_all.py (line DATA_ROOT = "data/").

Run the full pipeline:

bash
python src/run_all.py
This will:

Load and preprocess all nights (per‑subject normalisation)

Perform LOSO‑CV for Random Forest (and optionally other models)

Print classification metrics (accuracy, κ, F1 per stage)

Save the confusion matrix (results/fig2_rt_confusion.png)

Save Gini importance (results/fig6_feature_importance.png)

Save per‑subject κ boxplot (results/fig7_per_subject_kappa.png)

Results (reproduced)
Model	Accuracy	Cohen’s κ	Macro F1	Weighted F1
Random Forest	0.810	0.64	0.65	0.78
BiLSTM‑Attention	~0.58	0.19	0.35	0.37
SVM (RBF)	0.431	0.21	0.35	0.42
Logistic Reg.	0.267	0.12	0.29	0.31
Per‑stage F1 for Random Forest: Wake 0.53, N1 0.11, N2 0.86, N3 0.76, REM 0.84.

Citation
If you use this code, please cite the original paper (once published) and the PhysioNet dataset.

License
MIT
"""Generate all figures: confusion matrix, Gini importance, per-subject kappa."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, save_path="results/fig2_rt_confusion.png"):
    """Row-normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'})
    plt.ylabel('True Stage')
    plt.xlabel('Predicted Stage')
    plt.title('Random Forest Confusion Matrix (LOSO-CV, row-normalised)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_gini_importance(model, feature_names, save_path="results/fig6_feature_importance.png"):
    """Bar plot of Gini importance from trained Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Gini Importance (Mean Decrease in Node Impurity)')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_per_subject_kappa(kappas, save_path="results/fig7_per_subject_kappa.png"):
    """Boxplot of per-subject Cohen's κ across LOSO folds."""
    plt.figure(figsize=(6,8))
    plt.boxplot(kappas, vert=True, patch_artist=True)
    plt.ylabel("Cohen's κ")
    plt.title("Per‑Subject Performance (Random Forest, LOSO‑CV)")
    plt.xticks([1], ['All Subjects'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")
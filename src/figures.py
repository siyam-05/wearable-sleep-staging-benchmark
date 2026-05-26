python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def save_fig(fig, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_class_distribution(y_test, out_dir):
    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    counts = np.bincount(y_test)[:5]
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(classes, counts, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'])
    ax.set_ylabel('Number of epochs')
    ax.set_title('Class distribution in the test set')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f'{cnt}', ha='center')
    save_fig(fig, 'fig1_class_distribution.png', out_dir)

def plot_confusion_matrix(y_true, y_pred, out_dir):
    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Random Forest Confusion Matrix (row-normalised)')
    save_fig(fig, 'fig2_rf_confusion.png', out_dir)

def plot_model_comparison(results_df, out_dir):
    models = results_df.columns
    acc = results_df.loc['Accuracy']
    kappa = results_df.loc["Cohen's kappa"]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    rects1 = ax.bar(x - width/2, acc, width, label='Accuracy', color='teal')
    rects2 = ax.bar(x + width/2, kappa, width, label="Cohen's kappa", color='coral')
    ax.set_ylabel('Score')
    ax.set_title('Model comparison on test set')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    for rect in rects1+rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x()+rect.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', fontsize=8)
    save_fig(fig, 'fig3_model_comparison.png', out_dir)

def plot_per_class_performance(y_true, y_pred, out_dir):
    from sklearn.metrics import classification_report
    classes = ['Wake', 'N1', 'N2', 'N3', 'REM']
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    data = {c: [report[c][m] for m in metrics] for c in classes}
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10,6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i-1)*width, [data[c][i] for c in classes], width, label=metric.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0,1)
    ax.set_ylabel('Score')
    ax.set_title('Random Forest per‑stage performance')
    ax.legend()
    save_fig(fig, 'fig4_per_class.png', out_dir)

def plot_bilstm_overfitting(history, out_dir):
    epochs = range(1, len(history['accuracy'])+1)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(epochs, history['accuracy'], 'b-', label='Training accuracy')
    ax.plot(epochs, history['val_accuracy'], 'r--', label='Validation accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('BiLSTM‑Attention: overfitting')
    ax.legend()
    save_fig(fig, 'fig5_bilstm_overfitting.png', out_dir)

def plot_feature_importance(rf_model, out_dir):
    from features import FEATURE_NAMES
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(np.array(FEATURE_NAMES)[sorted_idx], importances[sorted_idx], color='teal')
    ax.set_xlabel('Gini importance')
    ax.set_title('Random Forest feature importance')
    save_fig(fig, 'fig6_feature_importance.png', out_dir)

def plot_per_subject_kappa(kappas, out_dir):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.boxplot(kappas, vert=False)
    ax.set_xlabel("Cohen's kappa")
    ax.set_title('Random Forest performance across 47 subjects (LOSO)')
    ax.set_xlim(0.3,0.9)
    save_fig(fig, 'fig7_per_subject_kappa.png', out_dir)
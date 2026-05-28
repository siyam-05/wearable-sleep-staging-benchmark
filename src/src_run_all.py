"""Main script: load data, run LOSO‑CV for RF, generate metrics and figures."""

import os
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from load_data import load_all_subjects
from train_eval import train_evaluate_loso
from figures import plot_confusion_matrix, plot_gini_importance, plot_per_subject_kappa

def main():
    # ========== CONFIGURATION ==========
    DATA_ROOT = r"E:\dataset_paper_1\Siyam\Downloads\bidsleep-dataset\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    # Change to your dataset path
    os.makedirs("results", exist_ok=True)

    print("Loading dataset...")
    data_dict = load_all_subjects(DATA_ROOT)
    n_subj = len(set(k.split('_')[0] for k in data_dict.keys()))
    print(f"Loaded {len(data_dict)} nights from {n_subj} subjects\n")

    # ========== Random Forest (main) ==========
    print("=== Random Forest (LOSO‑CV) ===")
    y_true, y_pred, kappas = train_evaluate_loso(data_dict, model_type='rf')

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc:.3f}")
    print(f"Cohen's κ: {kappa:.3f} (±{kappas.std():.3f})")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}\n")
    print("Per-stage classification report:")
    print(classification_report(y_true, y_pred, target_names=['Wake','N1','N2','N3','REM'],
                                labels=[0,1,2,3,4], zero_division=0))

    plot_confusion_matrix(y_true, y_pred, save_path="results/fig2_rt_confusion.png")
    plot_per_subject_kappa(kappas, save_path="results/fig7_per_subject_kappa.png")

    # ========== Gini importance on full dataset ==========
    print("\nTraining final Random Forest on all data for Gini importance...")
    X_all = np.vstack([feats for feats, _ in data_dict.values()])
    y_all = np.concatenate([lbls for _, lbls in data_dict.values()])
    final_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    final_rf.fit(X_all, y_all)
    feature_names = [
        'HR_mean', 'HR_std', 'HR_min', 'HR_max', 'HR_range', 'HR_delta', 'HR_missing',
        'Acc_X', 'Acc_Y', 'Acc_Z', 'Std_X', 'Std_Y', 'Std_Z',
        'SMA', 'Energy', 'Activity', 'Epoch_idx', 'Elapsed'
    ]
    plot_gini_importance(final_rf, feature_names, save_path="results/fig6_feature_importance.png")

    # ========== Optional: Comparison models (uncomment if desired) ==========
    # for mdl in ['svm', 'logreg', 'bilstm']:
    #     print(f"\n=== {mdl.upper()} ===")
    #     y_true_c, y_pred_c, _ = train_evaluate_loso(data_dict, model_type=mdl)
    #     print(f"Acc: {accuracy_score(y_true_c, y_pred_c):.3f}, "
    #           f"κ: {cohen_kappa_score(y_true_c, y_pred_c):.3f}")

    print("\nAll results saved in 'results/' folder.")

if __name__ == "__main__":
    main()
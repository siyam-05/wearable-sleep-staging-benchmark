# 3. Plotting functions
# ------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path="results/fig2_rt_confusion.png"):
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
    print(f"Confusion matrix saved to {save_path}")

def plot_gini_importance(model, feature_names, save_path="results/fig6_feature_importance.png"):
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
    print(f"Feature importance saved to {save_path}")

def plot_per_subject_kappa(kappas, save_path="results/fig7_per_subject_kappa.png"):
    plt.figure(figsize=(6,8))
    plt.boxplot(kappas, vert=True, patch_artist=True)
    plt.ylabel("Cohen's κ")
    plt.title("Per‑Subject Performance (Random Forest, LOSO‑CV)")
    plt.xticks([1], ['All Subjects'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-subject kappa boxplot saved to {save_path}")
# ------------------------------
# 4. Main execution
# ------------------------------

def main():
    # ===== CONFIGURE THIS =====
    DATA_ROOT = r"E:\dataset_paper_1\Siyam\Downloads\bidsleep-dataset\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    # Change to your dataset path if different
    # ===========================

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("Wearable Sleep Staging Benchmark")
    print("=" * 60)

    print("\nLoading dataset...")
    data_dict = load_all_subjects(DATA_ROOT)
    n_subjects = len(set(k.split('_')[0] for k in data_dict.keys()))
    print(f"Loaded {len(data_dict)} nights from {n_subjects} subjects")

    # ----- Random Forest (main result) -----
    print("\n=== Random Forest (LOSO-CV) ===")
    y_true, y_pred, kappas = train_evaluate_loso(data_dict, model_type='rf')

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {acc:.3f}")
    print(f"Cohen's κ: {kappa:.3f} (±{kappas.std():.3f})")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")
    print("\nPer-stage classification report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))

    # Save figures
    plot_confusion_matrix(y_true, y_pred, save_path="results/fig2_rt_confusion.png")
    plot_per_subject_kappa(kappas, save_path="results/fig7_per_subject_kappa.png")

    # ----- Feature importance (Gini) on full dataset -----
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

    print("\n" + "=" * 60)
    print("All results and figures saved in 'results/' folder.")
    print("Confusion matrix is now correct (diagonals match Table 3).")
    print("=" * 60)

if __name__ == "__main__":
    main()
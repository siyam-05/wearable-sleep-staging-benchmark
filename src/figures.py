# -------------------- Plotting --------------------
def plot_confusion_matrix(y_true, y_pred, save_path="results/fig2_rt_confusion.png"):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    labels = ['Wake','N1','N2','N3','REM']
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar_kws={'label':'Proportion'})
    plt.ylabel('True Stage'); plt.xlabel('Predicted Stage')
    plt.title('Random Forest Confusion Matrix (LOSO-CV, row-normalised)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_gini_importance(model, feature_names, save_path="results/fig6_feature_importance.png"):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(10,6))
    plt.barh(range(len(imp)), imp[idx], align='center')
    plt.yticks(range(len(imp)), [feature_names[i] for i in idx])
    plt.xlabel('Gini Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_subject_kappa(kappas, save_path="results/fig7_per_subject_kappa.png"):
    plt.figure(figsize=(6,8))
    plt.boxplot(kappas, vert=True, patch_artist=True)
    plt.ylabel("Cohen's κ"); plt.title("Per‑Subject Performance (RF, LOSO‑CV)")
    plt.xticks([1], ['All Subjects'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
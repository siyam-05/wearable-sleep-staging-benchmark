
# -------------------- Main --------------------
def main():
    DATA_ROOT = r"E:\dataset_paper_1\Siyam\Downloads\bidsleep-dataset\a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    os.makedirs("results", exist_ok=True)

    print("Loading dataset...")
    data_dict = load_all_subjects(DATA_ROOT)
    print(f"Loaded {len(data_dict)} nights from {len(set(k.split('_')[0] for k in data_dict.keys()))} subjects\n")

    print("=== Random Forest (LOSO‑CV) ===")
    y_true, y_pred, kappas = train_evaluate_loso(data_dict, model_type='rf')
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc:.3f}, Cohen's κ: {kappa:.3f} (±{kappas.std():.3f})")
    print(f"Macro F1: {macro_f1:.3f}, Weighted F1: {weighted_f1:.3f}\n")
    print("Per‑stage classification report:")
    print(classification_report(y_true, y_pred, target_names=['Wake','N1','N2','N3','REM']))

    plot_confusion_matrix(y_true, y_pred, save_path="results/fig2_rt_confusion.png")
    plot_per_subject_kappa(kappas, save_path="results/fig7_per_subject_kappa.png")

    # Gini importance on full data (not evaluation)
    print("\nTraining final RF on all data for Gini importance...")
    X_all = np.vstack([feats for feats, _ in data_dict.values()])
    y_all = np.concatenate([lbls for _, lbls in data_dict.values()])
    final_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    final_rf.fit(X_all, y_all)
    feature_names = ['HR_mean','HR_std','HR_min','HR_max','HR_range','HR_delta','HR_missing',
                     'Acc_X','Acc_Y','Acc_Z','Std_X','Std_Y','Std_Z','SMA','Energy','Activity',
                     'Epoch_idx','Elapsed']
    plot_gini_importance(final_rf, feature_names, save_path="results/fig6_feature_importance.png")
    print("\nAll results saved in 'results/' folder.")

# Run it
if __name__ == "__main__":
    main()


# -------------------- Data Loading Functions --------------------
def get_rec_start(mat, hr_df):
    rs = np.asarray(mat['recStart']).squeeze()
    if isinstance(rs, str):
        return pd.Timestamp(rs).timestamp()
    if isinstance(rs, (int, float, np.number)):
        if rs > 1e9:
            return float(rs)
        else:
            return (float(rs) - 719529) * 86400.0
    return float(hr_df['timestamp'].min()) - 3600

def load_one_night(night_dir):
    night_dir = Path(night_dir)
    mat = scipy.io.loadmat(str(night_dir / 'labels.mat'))
    if 'dreem_label' in mat:
        labels = mat['dreem_label'].squeeze().astype(int)
    elif 'dream_label' in mat:
        labels = mat['dream_label'].squeeze().astype(int)
    else:
        return None, None
    n_epochs = len(labels)

    hr = pd.read_csv(night_dir / 'hr.csv', header=None, names=['timestamp', 'bpm'])
    hr = hr.dropna().astype(float)
    if hr.empty:
        return None, None

    acc = pd.read_csv(night_dir / 'motion.csv')
    acc.columns = [c.strip().lower() for c in acc.columns]
    time_col = [c for c in acc.columns if 'time' in c][0]
    acc = acc[[time_col, 'x', 'y', 'z']]
    acc.columns = ['timestamp', 'x', 'y', 'z']
    acc = acc.dropna().astype(float)
    acc = acc.iloc[::5].reset_index(drop=True)

    rec_start = get_rec_start(mat, hr)
    hr['epoch'] = ((hr['timestamp'] - rec_start) // 30).astype(int)
    acc['epoch'] = ((acc['timestamp'] - rec_start) // 30).astype(int)
    hr = hr[(hr['epoch'] >= 0) & (hr['epoch'] < n_epochs)]
    acc = acc[(acc['epoch'] >= 0) & (acc['epoch'] < n_epochs)]
    if hr.empty:
        return None, None

    hr_group = hr.groupby('epoch')['bpm'].agg(['mean','std','min','max','count']).reindex(range(n_epochs), fill_value=0)
    hr_group = hr_group.fillna(0)
    hr_group['range'] = hr_group['max'] - hr_group['min']
    hr_group['delta'] = hr_group['mean'].diff().fillna(0)
    hr_group['missing'] = (hr_group['count'] == 0).astype(float)

    acc['sma'] = np.abs(acc['x']) + np.abs(acc['y']) + np.abs(acc['z'])
    acc['energy'] = acc['x']**2 + acc['y']**2 + acc['z']**2
    acc_mean = acc.groupby('epoch')[['x','y','z','sma','energy']].mean().reindex(range(n_epochs), fill_value=0)
    acc_std = acc.groupby('epoch')[['x','y','z']].std().reindex(range(n_epochs), fill_value=0).fillna(0)
    acc_mean.columns = ['acc_x','acc_y','acc_z','sma','energy']
    acc_std.columns = ['std_x','std_y','std_z']
    acc_feat = pd.concat([acc_mean, acc_std], axis=1)
    acc_feat['activity'] = (acc_feat['sma'] > 0.05).astype(float)

    temporal = np.column_stack([np.arange(n_epochs)/(n_epochs-1), np.arange(n_epochs)*0.5])
    features = np.column_stack([
        hr_group[['mean','std','min','max','range','delta','missing']].values,
        acc_feat[['acc_x','acc_y','acc_z','std_x','std_y','std_z','sma','energy','activity']].values,
        temporal
    ]).astype(np.float32)
    valid = (hr_group['count'] > 0).values
    return features[valid], labels[valid]

def load_all_subjects(data_root):
    data_root = Path(data_root)
    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
    DATA = {}
    for subj in tqdm(subjects, desc="Loading subjects"):
        nights = [d for d in subj.iterdir() if d.is_dir() and (d/'labels.mat').exists()]
        for night in nights:
            feats, lbls = load_one_night(night)
            if feats is not None and len(feats) > 10:
                mu, std = feats.mean(axis=0), feats.std(axis=0)
                std[std==0] = 1.0
                feats_norm = (feats - mu) / std
                DATA[f"{subj.name}_{night.name}"] = (feats_norm, lbls)
    return DATA

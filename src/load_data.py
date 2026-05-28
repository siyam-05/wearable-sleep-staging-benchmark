
---

## 4. `src/load_data.py`

```python
"""Data loading, feature extraction, and per-subject normalisation."""

import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
from tqdm import tqdm

def get_rec_start(mat, hr_df):
    """Parse recStart from .mat file (string, datenum, or numeric)."""
    rs = np.asarray(mat['recStart']).squeeze()
    if isinstance(rs, str):
        return pd.Timestamp(rs).timestamp()
    if isinstance(rs, (int, float, np.number)):
        if rs > 1e9:          # Unix timestamp
            return float(rs)
        else:                 # Excel/datenum
            return (float(rs) - 719529) * 86400.0
    # fallback: first HR timestamp minus 1 hour
    return float(hr_df['timestamp'].min()) - 3600

def load_one_night(night_dir):
    """
    Load one night, extract 18 features per 30‑s epoch.
    Returns (features, labels) or (None, None) if invalid.
    Labels are converted from 1..5 to 0..4.
    """
    night_dir = Path(night_dir)
    mat = scipy.io.loadmat(str(night_dir / 'labels.mat'))
    if 'dreem_label' in mat:
        labels = mat['dreem_label'].squeeze().astype(int)
    elif 'dream_label' in mat:
        labels = mat['dream_label'].squeeze().astype(int)
    else:
        return None, None
    # Convert labels from 1..5 to 0..4
    labels = labels - 1
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
    acc = acc.iloc[::5].reset_index(drop=True)   # 50Hz → 10Hz

    rec_start = get_rec_start(mat, hr)
    hr['epoch'] = ((hr['timestamp'] - rec_start) // 30).astype(int)
    acc['epoch'] = ((acc['timestamp'] - rec_start) // 30).astype(int)
    hr = hr[(hr['epoch'] >= 0) & (hr['epoch'] < n_epochs)]
    acc = acc[(acc['epoch'] >= 0) & (acc['epoch'] < n_epochs)]
    if hr.empty:
        return None, None

    # HR features (7)
    hr_group = hr.groupby('epoch')['bpm'].agg(['mean','std','min','max','count']) \
                 .reindex(range(n_epochs), fill_value=0)
    hr_group = hr_group.fillna(0)
    hr_group['range'] = hr_group['max'] - hr_group['min']
    hr_group['delta'] = hr_group['mean'].diff().fillna(0)
    hr_group['missing'] = (hr_group['count'] == 0).astype(float)

    # ACC features (11)
    acc['sma'] = np.abs(acc['x']) + np.abs(acc['y']) + np.abs(acc['z'])
    acc['energy'] = acc['x']**2 + acc['y']**2 + acc['z']**2
    acc_mean = acc.groupby('epoch')[['x','y','z','sma','energy']].mean() \
                  .reindex(range(n_epochs), fill_value=0)
    acc_std = acc.groupby('epoch')[['x','y','z']].std() \
                 .reindex(range(n_epochs), fill_value=0).fillna(0)
    acc_mean.columns = ['acc_x','acc_y','acc_z','sma','energy']
    acc_std.columns = ['std_x','std_y','std_z']
    acc_feat = pd.concat([acc_mean, acc_std], axis=1)
    acc_feat['activity'] = (acc_feat['sma'] > 0.05).astype(float)

    # Temporal features (2)
    temporal = np.column_stack([
        np.arange(n_epochs) / (n_epochs - 1),   # epoch_idx (0..1)
        np.arange(n_epochs) * 0.5               # elapsed minutes
    ])

    # Combine all 18 features
    features = np.column_stack([
        hr_group[['mean','std','min','max','range','delta','missing']].values,
        acc_feat[['acc_x','acc_y','acc_z','std_x','std_y','std_z','sma','energy','activity']].values,
        temporal
    ]).astype(np.float32)

    valid = (hr_group['count'] > 0).values
    return features[valid], labels[valid]

def load_all_subjects(data_root):
    """
    Load all nights, apply per-subject z-score normalisation.
    Returns dictionary: key = "subjectID_nightID", value = (features, labels).
    """
    data_root = Path(data_root)
    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
    data_dict = {}
    for subj in tqdm(subjects, desc="Loading subjects"):
        nights = [d for d in subj.iterdir() if d.is_dir() and (d/'labels.mat').exists()]
        for night in nights:
            feats, lbls = load_one_night(night)
            if feats is not None and len(feats) > 10:
                # Per-subject normalisation (mean/std computed only from this subject)
                mu = feats.mean(axis=0)
                std = feats.std(axis=0)
                std[std == 0] = 1.0
                feats_norm = (feats - mu) / std
                data_dict[f"{subj.name}_{night.name}"] = (feats_norm, lbls)
    return data_dict

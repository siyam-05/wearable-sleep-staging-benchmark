### 3. `src/load_data.py`

```python
import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
from tqdm import tqdm

def get_rec_start(mat, hr_df):
    """Parse recStart from mat file (string, datenum, or numeric)."""
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
    # Labels
    mat = scipy.io.loadmat(str(night_dir / 'labels.mat'))
    if 'dreem_label' in mat:
        labels = mat['dreem_label'].squeeze().astype(int)
    elif 'dream_label' in mat:
        labels = mat['dream_label'].squeeze().astype(int)
    else:
        return None, None
    # HR
    hr = pd.read_csv(night_dir / 'hr.csv', header=None, names=['timestamp', 'bpm'])
    hr = hr.dropna().astype(float)
    # ACC (downsample 50Hz->10Hz)
    acc = pd.read_csv(night_dir / 'motion.csv')
    acc.columns = [c.strip().lower() for c in acc.columns]
    time_col = [c for c in acc.columns if 'time' in c][0]
    x_col = [c for c in acc.columns if c == 'x'][0]
    y_col = [c for c in acc.columns if c == 'y'][0]
    z_col = [c for c in acc.columns if c == 'z'][0]
    acc = acc[[time_col, x_col, y_col, z_col]]
    acc.columns = ['timestamp', 'x', 'y', 'z']
    acc = acc.dropna().astype(float)
    acc = acc.iloc[::5].reset_index(drop=True)  # downsample
    # recStart
    rec_start = get_rec_start(mat, hr)
    # assign epochs (30s)
    n_epochs = len(labels)
    hr['epoch'] = ((hr['timestamp'] - rec_start) // 30).astype(int)
    acc['epoch'] = ((acc['timestamp'] - rec_start) // 30).astype(int)
    hr = hr[(hr['epoch'] >= 0) & (hr['epoch'] < n_epochs)]
    acc = acc[(acc['epoch'] >= 0) & (acc['epoch'] < n_epochs)]
    if len(hr) == 0:
        return None, None
    # Features
    hr_group = hr.groupby('epoch')['bpm'].agg(['mean','std','min','max','count']).reindex(range(n_epochs), fill_value=0)
    hr_group = hr_group.fillna(0)
    hr_group['range'] = hr_group['max'] - hr_group['min']
    hr_group['delta'] = hr_group['mean'].diff().fillna(0)
    hr_group['missing'] = (hr_group['count'] == 0).astype(float)
    # ACC
    acc['sma'] = np.abs(acc['x']) + np.abs(acc['y']) + np.abs(acc['z'])
    acc['energy'] = acc['x']**2 + acc['y']**2 + acc['z']**2
    acc_mean = acc.groupby('epoch')[['x','y','z','sma','energy']].mean().reindex(range(n_epochs), fill_value=0)
    acc_std = acc.groupby('epoch')[['x','y','z']].std().reindex(range(n_epochs), fill_value=0).fillna(0)
    acc_mean.columns = ['acc_x','acc_y','acc_z','sma','energy']
    acc_std.columns = ['std_x','std_y','std_z']
    acc_feat = pd.concat([acc_mean, acc_std], axis=1)
    acc_feat['activity'] = (acc_feat['sma'] > 0.05).astype(float)
    # temporal
    temporal = np.column_stack([np.arange(n_epochs)/(n_epochs-1), np.arange(n_epochs)*0.5])
    # combine
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
                # per-subject normalization
                mu, std = feats.mean(axis=0), feats.std(axis=0)
                std[std==0] = 1.0
                feats_norm = (feats - mu) / std
                DATA[f"{subj.name}_{night.name}"] = (feats_norm, lbls)
    return DATA
FEATURE_NAMES = [
    'HR_mean', 'HR_std', 'HR_min', 'HR_max', 'HR_range', 'HR_delta', 'HR_missing',
    'Acc_X', 'Acc_Y', 'Acc_Z', 'Std_X', 'Std_Y', 'Std_Z',
    'SMA', 'Energy', 'Activity', 'Epoch_idx', 'Elapsed'
]

FEATURE_TABLE_STR = r"""
| Feature name | Formula / Description | Signal source |
|--------------|----------------------|---------------|
| HR_mean | mean(bpm) over 30‑s epoch | Heart rate (1 Hz) |
| HR_std | standard deviation(bpm) | HR |
| HR_min | min(bpm) | HR |
| HR_max | max(bpm) | HR |
| HR_range | HR_max – HR_min | HR |
| HR_delta | HR_mean(epoch t) – HR_mean(epoch t-1) | HR |
| HR_missing | 1 if no HR samples in epoch, else 0 | HR |
| Acc_X | mean(x) over epoch | ACC (10 Hz after downsampling) |
| Acc_Y | mean(y) | ACC |
| Acc_Z | mean(z) | ACC |
| Std_X | standard deviation(x) | ACC |
| Std_Y | standard deviation(y) | ACC |
| Std_Z | standard deviation(z) | ACC |
| SMA | mean(|x|+|y|+|z|) | ACC |
| Energy | mean(x²+y²+z²) | ACC |
| Activity | 1 if SMA > 0.05 else 0 | ACC |
| Epoch_idx | normalized epoch index (epoch / total_epochs) | Derived |
| Elapsed | epoch number × 30 / 60 (minutes) | Derived |
"""
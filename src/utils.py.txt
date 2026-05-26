import numpy as np
import pandas as pd
from pathlib import Path

def save_results_table(results_dict, out_dir, filename):
    df = pd.DataFrame(results_dict)
    df.to_csv(Path(out_dir) / filename)

def set_seed(seed=42):
    import random, tensorflow as tf
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
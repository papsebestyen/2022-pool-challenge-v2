from scipy.spatial import cKDTree
import pickle
import pandas as pd
from constants import tree_path, data_path
from pathlib import Path
import numpy as np


pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]

df = (
    pd.read_csv(Path("data.csv"))
    .drop_duplicates(pos_cols)
    .dropna()
    .loc[:, lambda _df: _df.nunique() != 1]
)

position_data = df.loc[:, [*pos_cols]].values
position_data = np.append(position_data, np.zeros((position_data.shape[0], 1)), axis=1)

tree = cKDTree(position_data)

tree_path.write_bytes(pickle.dumps(tree))

data_path.write_bytes(pickle.dumps(df))

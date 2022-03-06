import pandas as pd
import numpy as np
import json
from pathlib import Path

pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
res_cols = ["msec", "subject", "trial"]


if __name__ == "__main__":
    df = pd.read_pickle("data.pkl")
    input_locations = json.loads(Path("input.json").read_text())
    out = []
    for query in input_locations:
        filt = (
            (df["subject"] == query["subject"])
            & (df["msec"] >= query["min_msec"])
            & (df["msec"] <= query["max_msec"])
        )
        pos_arr = np.array([query[c] for c in pos_cols], ndmin=2)
        min_ind = ((df.loc[filt, pos_cols] - pos_arr) ** 2).sum(axis=1).idxmin()
        out.append(df.loc[min_ind, res_cols].to_dict())

    Path("output.json").write_text(json.dumps(out))

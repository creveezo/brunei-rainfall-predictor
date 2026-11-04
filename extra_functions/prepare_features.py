import pandas as pd
import numpy as np


FEATURE_NAMES = [
    "sst_pc1", "sst_pc2", "sst_pc3",
    "sst_pc4", "sst_pc5", "sst_pc6",
    "month_sin", "month_cos"
]


def prepare_data(df):
    """
    Prepares input data for prediction.
    Expects target_month and six SST PCs.
    Keeps y_true if it is present, but never uses it as a feature.
    """

    required_columns = [
        "target_month",
        "sst_pc1", "sst_pc2", "sst_pc3",
        "sst_pc4", "sst_pc5", "sst_pc6"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    df = df.copy()

    df["target_month"] = pd.to_datetime(
        df["target_month"],
        format="%Y-%m"
    )

    df = df.sort_values("target_month").reset_index(drop=True)

    # Seasonal features
    month = df["target_month"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    X = df[FEATURE_NAMES].copy()

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Input data contains missing or infinite values."
        )

    return df, X


import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred):

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }


def create_comparison_df(df, predictions):

    result = pd.DataFrame({
        "target_month": df["target_month"].values,
        "y_true": df["y_true"].values,
        "y_pred": predictions
    })

    result["error"] = result["y_pred"] - result["y_true"]

    return result

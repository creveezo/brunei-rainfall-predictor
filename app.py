import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xarray as xr
from extra_functions.reconstruct_rainfall_map import plot_rainfall_map

# -------------------------
# 1. Page configuration
# -------------------------

st.set_page_config(
    page_title="Brunei Rainfall Predictor",
    page_icon="🌧️",
    layout="wide"
)

st.title("Brunei Rainfall Anomaly Predictor")
st.write(
    "Upload a CSV file containing ocean SST anomaly features "
    "to predict the rainfall anomaly PC1."
)

# -------------------------
# 2. Load models
# -------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")


@st.cache_resource
def load_eof_data(path):
    with xr.open_dataset(path) as ds:
        eof1 = ds["EOF"].isel(mode=0).load()
    return eof1


try:
    model = load_model()
    scaler = load_scaler()

except Exception as e:
    st.error(f"Could not load model or scaler: {e}")
    st.stop()


# -------------------------
# 3. Upload CSV
# -------------------------

feature_names = [
    'sst_pc1', 'sst_pc2', 'sst_pc3',
    'sst_pc4', 'sst_pc5', 'sst_pc6',
    'month_sin', 'month_cos'
]

uploaded_file = st.file_uploader(
    'Upload your CSV file. Required fields: "target_month" (datetime format yyyy-mm), "sst_pc1", "sst_pc2", ... "sst_pc6"',
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, index_col="target_month", parse_dates=True)

        st.subheader("Uploaded data")
        st.dataframe(df.head())

        # -------------------------
        # 4. Validate features
        # -------------------------



        if "month_sin" not in df.columns:
            month = df.index.month
            df["month_sin"] = np.sin(2 * np.pi * month / 12)

        if "month_cos" not in df.columns:
            month = df.index.month
            df["month_cos"] = np.cos(2 * np.pi * month / 12)

        missing_columns = [
            col for col in feature_names
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        X = df[feature_names].copy()

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            st.error("All model input features must be numeric.")
            st.stop()

        if X.isnull().any().any():
            st.error(
                "The input contains missing values. "
                "Please handle them before uploading."
            )
            st.stop()

        if not np.isfinite(X.to_numpy()).all():
            st.error("The input contains infinite values.")
            st.stop()



        # -------------------------
        # 5. Make predictions
        # -------------------------

        if st.button("Predict rainfall anomaly", type="primary"):

            if scaler is not None:
                X_processed = scaler.transform(X)
            else:
                X_processed = X

            predictions = model.predict(X_processed)

            # Save predictions between Streamlit reruns
            st.session_state["predictions"] = predictions
            st.session_state["results"] = X.copy()
            st.session_state["results"][
                "Predicted_Rainfall_PC1"
            ] = predictions
            st.session_state["prediction_dates"] = list(df.index)

        if "predictions" in st.session_state:
            dates = st.session_state["prediction_dates"]
            predictions = st.session_state["predictions"]
            results = st.session_state["results"]

            st.success("Prediction completed!")

            st.subheader("Prediction results")
            st.dataframe(results)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions",
                data=csv,
                file_name="rainfall_predictions.csv",
                mime="text/csv"
            )

            # Display timeline and map after prediction
            if "predictions" in st.session_state:

                dates = st.session_state["prediction_dates"]
                predictions = st.session_state["predictions"]

                st.subheader("Rainfall anomaly timeline")

                if len(dates) == 1:
                    selected_date = dates[0]
                    st.write(f"Selected month: **{selected_date:%B %Y}**")

                else:
                    selected_date = st.select_slider(
                        "Select a month",
                        options=dates,
                        value=dates[0],
                        format_func=lambda date: date.strftime("%b %Y")
                    )

                # Find prediction corresponding to selected month
                selected_index = dates.index(selected_date)
                selected_prediction = predictions[selected_index]

                st.metric(
                    "Predicted rainfall PC1",
                    f"{selected_prediction:+.3f}"
                )

                # Toggle numerical values on the map
                if "show_map_values" not in st.session_state:
                    st.session_state["show_map_values"] = False

                button_label = (
                    "Hide values on map"
                    if st.session_state["show_map_values"]
                    else "Show values on map"
                )

                if st.button(button_label):
                    st.session_state["show_map_values"] = (
                        not st.session_state["show_map_values"]
                    )

                # Reconstruct and display map
                eof1 = load_eof_data(
                    "data/mswep_brunei_landmask_EOF_patterns_10_fullgrid.nc"
                )

                anomaly = selected_prediction * eof1

                st.caption(
                    f"Spatial distribution of EOF1 contribution "
                    f"for {selected_date:%B %Y}"
                )

                plot_rainfall_map(
                    anomaly,
                    show_values=st.session_state["show_map_values"]
                )


    except Exception as e:
        st.error(f"An error occurred: {e}")
        try:
            df = pd.read_csv(uploaded_file, usecols=['target_month'])
            st.error(f"An error occurred: {e}")
        except ValueError:
            st.error('An error occurred: "target month" is a required field.')

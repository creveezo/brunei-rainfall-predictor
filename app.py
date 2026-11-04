import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
import joblib
import traceback

from extra_functions.reconstruct_rainfall_map import plot_rainfall_map
from extra_functions.prepare_features import prepare_data
from extra_functions.evaluate_model import (
    evaluate_predictions,
    create_comparison_df
)

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Brunei Rainfall Predictor",
    page_icon="🌧️",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

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

# ============================================================
# PREDICTION
# ============================================================

def make_predictions(X):
    model = load_model()
    scaler = load_scaler()

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    return predictions


# ============================================================
# VISUALIZATION
# ============================================================

def show_prediction_map(pc_value, eof1, title):
    anomaly = pc_value * eof1

    st.subheader(title)
    plot_rainfall_map(anomaly)


def show_demo_results(df, predictions):
    """
    Displays model evaluation and comparison with actual values.
    """

    comparison = create_comparison_df(df, predictions)

    # Metrics
    metrics = evaluate_predictions(
        comparison["y_true"],
        comparison["y_pred"]
    )

    st.subheader("Model performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", f"{metrics['RMSE']:.3f}")
    col2.metric("MAE", f"{metrics['MAE']:.3f}")
    col3.metric("R²", f"{metrics['R²']:.3f}")

    # Table
    st.subheader("Predicted vs actual values")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.download_button(
        "Download comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="rainfall_predictions.csv",
        mime="text/csv"
    )

    # Time series
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=comparison["target_month"],
        y=comparison["y_true"],
        mode="lines+markers",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=comparison["target_month"],
        y=comparison["y_pred"],
        mode="lines+markers",
        name="Predicted"
    ))

    fig.update_layout(
        title="Actual vs predicted rainfall PC1",
        xaxis_title="Month",
        yaxis_title="Rainfall PC1",
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    return comparison


# ============================================================
# MAIN INTERFACE
# ============================================================

st.title("Brunei Rainfall Predictor")

st.write(
    "Predict rainfall anomalies in Brunei using "
    "sea surface temperature patterns."
)

mode = st.radio(
    "Choose a mode:",
    ["Upload my data", "Explore demo"],
    horizontal=True
)


# ============================================================
# 1. DATA SOURCE — MODE-DEPENDENT INPUT
# ============================================================

df = None

if mode == "Upload my data":
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

else:
    st.write(
        "Explore the model using the held-out test dataset."
    )

    if st.button("Load demo data"):
        st.session_state["demo_loaded"] = True

    if st.session_state.get("demo_loaded", False):
        df = pd.read_csv("data/X_test.csv")

# ============================================================
# 2. SHARED PIPELINE — PREPARATION AND PREDICTION
# ============================================================

if df is not None:
    try:
        prepared_df, X = prepare_data(df)

        st.subheader("Input data")
        st.dataframe(prepared_df, use_container_width=True)

        if st.button("Run prediction"):
            predictions = make_predictions(X)

            st.session_state["results"] = {
                "mode": mode,
                "df": prepared_df,
                "predictions": predictions
            }

    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())


# ============================================================
# 3. RESULTS — MODE-DEPENDENT OUTPUT
# ============================================================

results = st.session_state.get("results")

if results is not None and results["mode"] == mode:

    result_df = results["df"].copy()
    predictions = results["predictions"]

    # Load EOF data
    eof1 = load_eof_data(
        "data/mswep_brunei_landmask_EOF_patterns_10_fullgrid.nc"
    )

    # Prepare common prediction results
    dates = pd.to_datetime(result_df["target_month"]).tolist()

    prediction_table = pd.DataFrame({
        "target_month": dates,
        "y_pred": predictions
    })

    st.divider()
    st.header("Results")

    # ========================================================
    # DEMO MODE
    # ========================================================

    if mode == "Explore demo":

        # Load actual values from a separate file
        y_true_df = pd.read_csv("data/y_true.csv")

        if not {"target_month", "y_true"}.issubset(y_true_df.columns):
            st.error(
                "y_true.csv must contain target_month and y_true columns."
            )
            st.stop()

        y_true_df["target_month"] = pd.to_datetime(
            y_true_df["target_month"],
            format="%Y-%m"
        )

        # Merge predictions and actual values by date
        comparison = prediction_table.merge(
            y_true_df[["target_month", "y_true"]],
            on="target_month",
            how="inner",
            validate="one_to_one"
        )

        if comparison.empty:
            st.error(
                "No matching dates found between demo_test.csv and y_true.csv."
            )
            st.stop()

        if len(comparison) != len(prediction_table):
            st.warning(
                "Some prediction dates do not have matching actual values. "
                "Only matching months are included in the comparison."
            )

        comparison["error"] = (
            comparison["y_pred"] - comparison["y_true"]
        )

        # ----------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------

        metrics = evaluate_predictions(
            comparison["y_true"],
            comparison["y_pred"]
        )

        st.subheader("Model performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("RMSE", f"{metrics['RMSE']:.3f}")
        col2.metric("MAE", f"{metrics['MAE']:.3f}")
        col3.metric("R²", f"{metrics['R²']:.3f}")

        # ----------------------------------------------------
        # Comparison table
        # ----------------------------------------------------

        st.subheader("Predicted vs actual values")

        st.dataframe(
            comparison,
            use_container_width=True
        )

        # Save table in session state
        st.session_state["comparison_table"] = comparison

        # Download results
        st.download_button(
            label="Download results CSV",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="rainfall_prediction_results.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # Timeline and map
        # ----------------------------------------------------

        st.divider()
        st.subheader("Rainfall anomaly timeline")

        if len(comparison) == 1:
            selected_date = comparison.iloc[0]["target_month"]
            st.write(
                f"Selected month: **{selected_date:%B %Y}**"
            )

        else:
            selected_date = st.select_slider(
                "Select a month",
                options=comparison["target_month"].tolist(),
                value=comparison.iloc[0]["target_month"],
                format_func=lambda date: date.strftime("%b %Y")
            )

        selected = comparison[
            comparison["target_month"] == selected_date
        ].iloc[0]

        # Show selected month metrics
        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted rainfall PC1",
            f"{selected['y_pred']:+.3f}"
        )

        col2.metric(
            "Actual rainfall PC1",
            f"{selected['y_true']:+.3f}"
        )

        # Toggle numerical values on maps
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

        # Display predicted and actual maps
        st.caption(
            f"Spatial distribution of EOF1 contribution "
            f"for {selected_date:%B %Y}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Predicted rainfall anomaly**")

            predicted_anomaly = selected["y_pred"] * eof1

            plot_rainfall_map(
                predicted_anomaly,
                show_values=st.session_state["show_map_values"]
            )

        with col2:
            st.markdown("**Actual rainfall anomaly**")

            actual_anomaly = selected["y_true"] * eof1

            plot_rainfall_map(
                actual_anomaly,
                show_values=st.session_state["show_map_values"]
            )


        # ----------------------------------------------------
        # Actual vs predicted timeline
        # ----------------------------------------------------

        st.divider()
        st.subheader("Actual vs predicted rainfall anomaly")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=comparison["target_month"],
            y=comparison["y_true"],
            mode="lines+markers",
            name="Actual (y_true)",
            line=dict(width=2),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=comparison["target_month"],
            y=comparison["y_pred"],
            mode="lines+markers",
            name="Predicted (y_pred)",
            line=dict(width=2, dash="dash"),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="Actual vs predicted rainfall PC1",
            xaxis_title="Month",
            yaxis_title="Rainfall PC1",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.update_xaxes(
            tickformat="%b %Y",
            showgrid=True
        )

        fig.update_yaxes(
            zeroline=True,
            zerolinewidth=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # UPLOAD MY DATA MODE
    # ========================================================

    else:

        st.subheader("Predicted rainfall PC1")

        st.dataframe(
            prediction_table,
            use_container_width=True
        )

        # Save predictions table in session state
        st.session_state["prediction_table"] = prediction_table

        # Download predictions
        st.download_button(
            label="Download predictions CSV",
            data=prediction_table.to_csv(index=False).encode("utf-8"),
            file_name="rainfall_predictions.csv",
            mime="text/csv"
        )

        # ----------------------------------------------------
        # Timeline and map — same interface as before
        # ----------------------------------------------------

        st.subheader("Rainfall anomaly timeline")

        if len(dates) == 1:
            selected_date = dates[0]
            st.write(
                f"Selected month: **{selected_date:%B %Y}**"
            )

        else:
            selected_date = st.select_slider(
                "Select a month",
                options=dates,
                value=dates[0],
                format_func=lambda date: date.strftime("%b %Y")
            )

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
        anomaly = selected_prediction * eof1

        st.caption(
            f"Spatial distribution of EOF1 contribution "
            f"for {selected_date:%B %Y}"
        )

        plot_rainfall_map(
            anomaly,
            show_values=st.session_state["show_map_values"]
        )

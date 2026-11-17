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
from extra_functions.datasets_visualisation import show_dataset_explorer

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

tab_data, tab_prediction = st.tabs([
    "📊 Data Explorer",
    "🌧️ Rainfall Prediction"
])

with tab_data:

    # 1. PROJECT OVERVIEW
    st.title("Data Explorer")
    st.write(
        "Explore the climate datasets, data processing methodology, "
        "and statistical patterns behind the Brunei Rainfall Predictor."
    )

    st.markdown("""
    ### Research Objective

    This project investigates the relationship between sea surface
    temperature (SST) variability across the Indian and Pacific Oceans
    and precipitation variability in Brunei Darussalam.

    The main objective is to identify large-scale oceanic patterns
    that may be associated with changes in regional precipitation
    and explore whether these patterns influence
    precipitation anomalies in Brunei.
    """)

    st.divider()


    # 2. DATASETS
    st.header("Datasets")

    st.markdown("""
    This project combines two climate datasets:""")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
    
        ### 2.1. MSWEP (Multi-Source Weighted-Ensemble Precipitation)
    
        The precipitation dataset provides information about rainfall over the Brunei region.
    
        The original daily observations are processed into monthly
        precipitation totals. The resulting dataset is used to investigate
        temporal and spatial precipitation variability.
    
        **Main characteristics:**
        - Variable: Precipitation
        - Spatial coverage: Brunei region
        - Resolution: 15x13 grid points
        - Units: mm/month
        - Time period: Sep 1981 - Dec 2025
        """)

    with col2:
        st.markdown("""
    
        ### 2.2. SST (Sea Surface Temperature Dataset)
    
        The SST dataset contains sea surface temperature observations
        across the Indian and Pacific Oceans.
    
        It is used to identify large-scale oceanic variability patterns
        that may influence precipitation in Brunei.
    
        **Main characteristics:**
        - Variable: Sea surface temperature (SST)
        - Spatial coverage: Indian and Pacific Oceans
        - Resolution: 259,200 grid points
        - Time period: Sep 1981 - Dec 2025
        - Units: °C
        """)

    show_dataset_explorer()
    st.markdown("This visualization shows only a 12-month sample of the full dataset.")
    st.divider()


    # ============================================================
    # 3. DATA PROCESSING
    # ============================================================

    st.header("Data Processing")

    st.markdown("""
    Before performing statistical analysis, both datasets undergo
    several preprocessing steps.

    ### Precipitation Processing

    1. **Spatial filtering** — retain grid cells belonging to
       the Brunei study region using a land mask.
    2. **Monthly aggregation** — calculate monthly precipitation
       totals from daily observations.
    3. **Monthly climatology** — calculate the average precipitation
       for each calendar month across the available years.
    4. **Anomaly calculation** — subtract the corresponding
       monthly climatology from each observation.
    5. **Detrending** — remove the linear temporal trend
       from each grid cell.
    6. **Area weighting** — apply latitude-based weighting
       to account for differences in grid cell area.

    ### SST Processing

    1. **Spatial selection** — retain the selected Indian and
       Pacific Ocean region.
    2. **Monthly climatology** — calculate the average precipitation
       for each calendar month across the available years.
    3. **Anomaly calculation** — remove the seasonal climatology
       to focus on deviations from typical SST conditions.
    4. **Spatial weighting** — account for differences in
      grid cell area.
    5. **EOF analysis** — extract the dominant patterns
       of SST variability.
    """)

    st.divider()

    # ============================================================
    # 4. WHAT ARE EOFs AND PCs?
    # ============================================================

    st.header("Understanding EOFs and Principal Components")

    st.markdown("""
    ### What is EOF Analysis?

    Empirical Orthogonal Function (EOF) analysis is a statistical
    technique used to identify dominant patterns of variability
    in multidimensional climate data.

    Climate datasets often contain thousands of spatial grid cells
    observed over many years. An EOF analysis reduces this complex
    information into a smaller number of spatial patterns and
    corresponding time series.

    In this project, EOF analysis is applied to identify the
    dominant modes of precipitation and SST variability.

    ### How Does EOF Analysis Work?

    Mathematically, the decomposition can be expressed as:

    """)

    st.latex(r"X = U \Sigma V^T")

    st.markdown("""
    Where:

    - **X** — the original data matrix.
    - **U** — contains the temporal information.
    - **Σ** — contains singular values describing the strength
      of each mode.
    - **V** — contains the spatial patterns.

    ### What are EOFs?

    EOFs (Empirical Orthogonal Functions) represent spatial patterns
    of variability.

    Each EOF describes how different geographical locations
    contribute to a particular mode of variation.

    For example, an EOF map may show whether precipitation
    anomalies tend to vary in the same direction across Brunei
    or whether different parts of the region exhibit contrasting
    patterns.

    ### What are PCs?

    PCs (Principal Components) are time series associated with
    the corresponding EOF patterns.

    A PC describes how strongly a particular spatial pattern
    is expressed at each time step.

    For example:
    - A positive PC value indicates a positive expression
      of the corresponding EOF pattern.
    - A negative PC value indicates an expression in the
      opposite direction.
    - A value close to zero indicates a weak expression
      of that mode.

    Each EOF has a corresponding PC with the same mode number:
    EOF1 corresponds to PC1, EOF2 to PC2, and so on.

    ### Explained Variance

    The explained variance indicates how much of the total
    variability is represented by each EOF mode.

    The first EOF usually captures the largest share of variance,
    followed by the second EOF, and so on.
    """)

    st.latex(
        r"\text{Explained Variance}_k = "
        r"\frac{\sigma_k^2}{\sum_i \sigma_i^2} \times 100\%"
    )

    # PLACEHOLDER:
    # Explained variance bar chart:
    # X-axis: EOF mode
    # Y-axis: Explained variance (%)
    #
    # PLACEHOLDER:
    # Cumulative explained variance line chart:
    # X-axis: Number of EOF modes
    # Y-axis: Cumulative explained variance (%)


    st.divider()


    # ============================================================
    # 5. PRECIPITATION EOF ANALYSIS
    # ============================================================

    st.header("Precipitation EOF Analysis")

    st.markdown("""
    EOF analysis is applied to the processed precipitation anomalies
    over the Brunei study region.

    The resulting EOF patterns describe the dominant spatial
    structures of precipitation variability, while the corresponding
    PCs show how these patterns evolve over time.
    """)

    # PLACEHOLDER:
    # Interactive EOF selector:
    # Select EOF mode (EOF1–EOF10)

    # PLACEHOLDER:
    # Selected precipitation EOF spatial map:
    # Color scale: EOF loading
    # Title: "Precipitation EOF [number]"

    # PLACEHOLDER:
    # Corresponding precipitation PC time series:
    # X-axis: Year
    # Y-axis: Principal component amplitude

    # PLACEHOLDER:
    # Precipitation explained variance chart

    st.subheader("EOF1 Reconstruction")

    st.markdown("""
    The first EOF mode can be used to reconstruct the contribution
    of the dominant precipitation variability pattern at a selected
    time step.

    This reconstruction represents only the contribution of EOF1,
    rather than the complete precipitation anomaly field.
    """)

    st.latex(r"P'_1(x,y,t) = PC_1(t) \cdot EOF_1(x,y)")

    # PLACEHOLDER:
    # Interactive EOF1 reconstruction map:
    # Select a month
    # Display PC1(t) × EOF1(x, y)


    st.divider()


    # ============================================================
    # 6. SST EOF ANALYSIS
    # ============================================================

    st.header("Sea Surface Temperature EOF Analysis")

    st.markdown("""
    EOF analysis is also applied to SST anomalies across the
    selected Indian and Pacific Ocean region.

    The resulting SST EOFs represent dominant oceanic variability
    patterns. Their corresponding PCs provide time series that
    describe the temporal evolution of these patterns.

    These SST principal components are used as potential predictors
    in the precipitation prediction model.
    """)

    # PLACEHOLDER:
    # SST EOF selector:
    # Select EOF mode (EOF1–EOF6 or more, depending on saved results)

    # PLACEHOLDER:
    # Selected SST EOF spatial map:
    # Color scale: SST EOF loading

    # PLACEHOLDER:
    # Corresponding SST PC time series

    # PLACEHOLDER:
    # SST explained variance chart

    # PLACEHOLDER:
    # SST PC1–PC6 overview:
    # Six time series charts (2 columns × 3 rows)


    st.divider()


    # ============================================================
    # 7. SST–PRECIPITATION RELATIONSHIPS
    # ============================================================

    st.header("SST–Precipitation Relationships")

    st.markdown("""
    This section explores statistical relationships between
    large-scale oceanic variability and precipitation variability
    in Brunei.

    The analysis focuses on the relationship between SST principal
    components and precipitation principal components.
    """)

    st.subheader("Correlation Analysis")

    st.markdown("""
    Correlation analysis is used to measure the strength and
    direction of linear relationships between SST PCs and
    precipitation PCs.
    """)

    # PLACEHOLDER:
    # Correlation matrix heatmap:
    # Rows: SST PCs
    # Columns: Precipitation PCs
    # Values: Pearson correlation coefficient (r)

    # PLACEHOLDER:
    # Scatter plots:
    # X-axis: SST PC
    # Y-axis: Precipitation PC1
    # Include regression line and correlation coefficient.

    st.subheader("Lagged Correlation Analysis")

    st.markdown("""
    Oceanic variability may precede changes in regional precipitation.
    Therefore, correlations are also examined at different time lags.

    A positive lag indicates that SST variability occurs before
    the corresponding precipitation variability.

    For example:
    - Lag 0: SST and precipitation in the same month.
    - Lag 1: SST one month before precipitation.
    - Lag 2: SST two months before precipitation.
    - Lag 3: SST three months before precipitation.
    """)

    # PLACEHOLDER:
    # Lagged correlation chart:
    # X-axis: Lag (months)
    # Y-axis: Pearson correlation coefficient (r)
    # Separate lines for different SST PCs.

    # PLACEHOLDER:
    # Rolling correlation chart:
    # X-axis: Year
    # Y-axis: Rolling Pearson correlation coefficient (r)
    # Window: [ADD WINDOW SIZE]


    st.divider()


    # ============================================================
    # 8. KEY FINDINGS
    # ============================================================

    st.header("Key Findings")

    st.markdown("""
    This section will summarize the main observations obtained
    from the exploratory analysis.

    """)

    # PLACEHOLDER:
    # Automatically generated or manually written findings:
    # - Dominant precipitation variability pattern
    # - Dominant SST variability patterns
    # - Strongest SST–precipitation relationships
    # - Most relevant time lags
    # - Potential implications for precipitation prediction

    st.info(
        "The visualizations and statistical results presented here "
        "are intended for exploratory analysis. Correlation does not "
        "necessarily imply causation."
    )
























with tab_prediction:
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

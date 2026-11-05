# Brunei Rainfall Predictor

**An interactive machine learning application for predicting monthly precipitation anomalies in Brunei using sea surface temperature (SST) patterns in Indo-Pacific region.**

[Presentation slides](https://docs.google.com/presentation/d/1o7RIvhFlEYOCF7T0qPuWmuF15TgKgzIOC_xi7yb0e5k/edit?usp=sharing)
[Application](https://brunei-rainfall-predictor.streamlit.app/)

---

## Overview

Brunei Rainfall Predictor is a machine learning project that explores the relationship between sea surface temperature variability in the Indian and Pacific Oceans and precipitation anomalies in Brunei.

The project aims to investigate whether large-scale ocean temperature patterns can provide useful information for predicting local precipitation variability.

The application uses principal components derived from SST data, together with seasonal features, to predict the first principal component (PC1) of Brunei precipitation anomalies.

The project combines climate data preprocessing, statistical analysis, dimensionality reduction, machine learning, and interactive data visualization.

### Main objectives

- Process and analyze historical precipitation and sea surface temperature data.
- Identify major spatial patterns of climate variability using Empirical Orthogonal Function (EOF) analysis.
- Investigate relationships between oceanic SST variability and precipitation anomalies in Brunei.
- Develop a machine learning model for predicting precipitation PC1.
- Provide an interactive interface for exploring data and visualizing model predictions.

---

## Live Demo

**Streamlit application:** [Open the application](https://brunei-rainfall-predictor.streamlit.app/)

The application currently provides two modes:

- **Upload my data:** Upload a CSV file containing the required input features and generate predictions.
- **Explore demo:** Explore predictions on the held-out test dataset and compare them with observed values.

The application includes:

- Rainfall PC1 predictions.
- Interactive spatial maps of reconstructed precipitation anomalies.
- Actual vs. predicted time-series comparison.
- Model evaluation metrics.
- Downloadable prediction results.

*An interactive data exploration dashboard is planned as a future improvement.*

---

## Research Question

**Can sea surface temperature variability patterns in the Indian and Pacific Oceans help predict monthly precipitation anomalies in Brunei?**

The project focuses on representing large-scale oceanic variability through a relatively small number of principal components and examining their relationship with local precipitation patterns.

The prediction target is the first principal component of Brunei precipitation anomalies.

---

## Data Sources

The project uses historical climate datasets representing precipitation over Brunei and sea surface temperatures across the Indian and Pacific Oceans.

### 1. Precipitation Data

**Dataset:** MSWEP (Multi-Source Weighted-Ensemble Precipitation)

The preprocessing workflow includes:

1. Reading daily precipitation data.
2. Applying a Brunei land mask.
3. Aggregating daily precipitation observations into monthly values.
4. Calculating calendar-month climatology.
5. Computing monthly precipitation anomalies.
6. Removing linear trends.
7. Applying latitude-based area weighting.
8. Performing EOF analysis using Singular Value Decomposition (SVD).

The precipitation analysis is restricted to grid cells within the Brunei land mask.

**Dataset details:**

| Property | Description |
|---|---|
| Dataset | MSWEP |
| Variable | Precipitation |
| Analysis region | Brunei |
| Temporal coverage | 571 months |
| Spatial resolution | 15x13 grid points |
| Anomaly units | mm/month |

### 2. Sea Surface Temperature Data

The preprocessing workflow includes:

1. Reading SST.
2. Removing the linear trend at each valid SST grid cell.
3. Computing monthly climatological means.
4. Computing monthly anomalies by removing those climatological means.
5. Applying latitude-based area weighting.
6. Performing EOF analysis using Singular Value Decomposition (SVD).

**Dataset details:**

| Property | Description |
|---|---|
| Variable | Sea Surface Temperature (SST) |
| Study regions | Indian and Pacific Oceans |
| Temporal coverage | 539 months |
| Spatial resolution | 259,200 grid points |
| Number of retained components | 6 |

---

## Methodology

### 1. Data Preprocessing

Stages of data preprocessing for both datasets are described above.

### 2. EOF Analysis and Dimensionality Reduction

Empirical Orthogonal Function (EOF) analysis is used to identify dominant spatial patterns of variability in the climate datasets.

The decomposition is performed using Singular Value Decomposition (SVD).

The analysis produces:

- **EOF patterns:** Spatial structures representing dominant modes of variability.
- **Principal Components (PCs):** Time series describing how strongly each spatial pattern is expressed over time.
- **Explained variance:** The proportion of total variability represented by each mode.

The EOF mode can be multiplied by the corresponding PC1 value to reconstruct the spatial contribution of that mode to the precipitation anomaly field.


### 3. Feature Engineering

The current model input consists of:

| Feature | Description |
|---|---|
| `sst_pc1` | First SST principal component |
| `sst_pc2` | Second SST principal component |
| `sst_pc3` | Third SST principal component |
| `sst_pc4` | Fourth SST principal component |
| `sst_pc5` | Fifth SST principal component |
| `sst_pc6` | Sixth SST principal component |
| `month_sin` | Sine transformation of the target month |
| `month_cos` | Cosine transformation of the target month |

The prediction target is precipitation MSWEP PC1.

### 4. Machine Learning Model

The input features are standardized using a fitted scaler.

**Model architecture:** Random Forest

**Training configuration:** 

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 20| 
| `min_samples_split` | 2 |
| `min_samples_leaf` | 4 | 
| `max_features` | 0.7 |

### 5. Model Evaluation

Model performance is evaluated on a held-out test dataset. The current evaluation setup uses the final 60 months (2021–2025) as the test period.

The following metrics are used:

| Metric | Result |
|---|---|
| RMSE | 621.434 |
| MAE | 507.189 |
| R² | 0.071 |

---

## Notebooks

`notebooks` directory contains python notebooks with the code, preprocessing each nc dataset, turning it into csv dataset and model training. 

---

## Application

The application is built with Streamlit and provides an interactive interface for generating and exploring precipitation predictions.

### Prediction workflow

1. Select the input mode.
2. Upload a CSV file or select the demo dataset.
3. The application validates and prepares the input features.
4. The saved scaler transforms the features.
5. The trained model generates precipitation PC1 predictions.
6. The application displays prediction results and visualizations.

### Input CSV format

For custom predictions, the uploaded CSV file must contain the following columns:

```csv
target_month,sst_pc1,sst_pc2,sst_pc3,sst_pc4,sst_pc5,sst_pc6
2025-01-01,0.12,-0.35,0.21,0.08,-0.14,0.31
2025-02-01,0.18,-0.22,0.16,0.11,-0.09,0.27
```

The example values above are illustrative and do not represent actual observations.

The application derives `month_sin` and `month_cos` from `target_month`.

### Visualizations

The application includes:

- Interactive maps of reconstructed precipitation anomalies.
- Numerical values displayed on maps (optional).
- Actual vs. predicted precipitation PC1 time series.
- Comparison tables containing observed values, predictions, and prediction errors.

---

## Project Structure

```text
brunei-rainfall-predictor/
│
├── data/
│   ├── X_test.csv
│   ├── y_true.csv
│   ├── mswep_brunei_landmask_EOF_patterns_10_fullgrid.nc
│   ├── brunei.geojson
│   └── malaysia.geojson
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── extra_functions/
│   ├── evaluate_model.py
│   ├── prepare_features.py
│   └── reconstruct_rainfall_map.py
│ 
├── notebooks/
│   ├── MSWEP_preprocessing.ipynb
│   ├── SST_preprocessing.ipynb
│   ├── dataset_creation.ipynb
│   └── training_and_evaluating_models.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```
---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/brunei-rainfall-predictor.git

cd brunei-rainfall-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

## Limitations

Several limitations should be considered when interpreting the results:

- Climate variability is influenced by multiple interacting atmospheric and oceanic processes that may not be fully represented by the selected predictors.
- Model performance is evaluated on a specific historical test period and may not generalize equally well to other climatic periods.
- The quality of the results depends on the completeness, consistency, and temporal alignment of the underlying datasets.

---

## Future Improvements

Planned improvements include:

- Adding a dedicated data exploration dashboard.
- Providing more detailed descriptive statistics and visualizations of precipitation and SST datasets.
- Exploring alternative feature selection and dimensionality reduction approaches.

---

## Technologies

The project uses the following tools and libraries:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation |
| Geopandas | Map visualization |
| Xarray | Multidimensional climate data processing |
| SciPy | Scientific computing and signal processing |
| Scikit-learn | Machine learning and feature scaling |
| Joblib | Saving and loading model artifacts |
| Plotly | Interactive visualizations |
| Streamlit | Web application framework |
| NetCDF | Climate data storage format |


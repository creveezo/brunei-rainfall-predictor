# Brunei Rainfall Predictor

**An interactive machine learning application for predicting monthly precipitation anomalies in Brunei using sea surface temperature (SST) patterns in the    Indo-Pacific region.**

[Presentation slides](https://docs.google.com/presentation/d/1o7RIvhFlEYOCF7T0qPuWmuF15TgKgzIOC_xi7yb0e5k/edit?usp=sharing)
[Application](https://brunei-rainfall-predictor.streamlit.app/)

---

## Overview

Brunei Rainfall Predictor is a machine learning project that explores the relationship between sea surface temperature variability in the Indian and Pacific Oceans and precipitation anomalies in Brunei.

The project aims to investigate whether large-scale ocean temperature patterns can provide useful information for predicting local precipitation variability.

The application uses principal components derived from SST data, together with seasonal features, to predict the first principal component (PC1) of Brunei precipitation anomalies.

The project combines climate data preprocessing, statistical analysis, dimensionality reduction, machine learning, and interactive data visualization.

### The aim of the project

The aim of this project is to use Indo-Pacific sea surface temperature (SST) anomalies to predict monthly precipitation anomalies in Brunei.

### Main objectives

- Process historical precipitation and sea surface temperature data.
- Identify major spatial patterns of climate variability using Empirical Orthogonal Function (EOF) analysis.
- Develop and evaluate machine learning models to predict the first principal component (PC1) of Brunei precipitation anomalies.
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

The **Data Explorer** includes sample maps, EOF patterns, principal component time series, explained-variance charts, and SST–precipitation correlations. Use the sidebar to switch between Data Explorer and Rainfall Prediction.

---

## Research Question

**Can sea surface temperature variability patterns in the Indian and Pacific Oceans help predict monthly precipitation anomalies in Brunei?**

The project focuses on representing large-scale oceanic variability through a relatively small number of principal components and examining their relationship with local precipitation patterns.

The prediction target is the first principal component (PC1) of the Brunei precipitation anomaly field. Precipitation anomalies are calculated relative to the climatological mean for the corresponding calendar month. For example, September precipitation is compared with the long-term average for September, rather than with the annual average. EOF analysis is then used to identify the dominant spatial patterns of precipitation variability. The first principal component represents the temporal evolution of the leading spatial pattern. 

The first EOF mode explains 86.7% of the total variance in the precipitation anomaly field, which motivates using PC1 as the prediction target.

---

## Datasets

The project uses historical climate datasets representing precipitation over Brunei and sea surface temperatures across the Indian and Pacific Oceans.

### 1. Precipitation Data

**Dataset:** MSWEP (Multi-Source Weighted-Ensemble Precipitation)

The preprocessing workflow includes:

1. Reading daily precipitation data.
2. Applying a Brunei land mask.
3. Aggregating daily precipitation observations into monthly values.
4. Removing linear trends.
5. Computing monthly precipitation anomalies by removing calendar-month climatology.
6. Applying latitude-based area weighting.
7. Performing EOF analysis using Singular Value Decomposition (SVD).

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
| Grid dimensions | 259,200 grid points |
| Number of retained components | 6 |

Since the precipitation and SST datasets have different temporal coverage, they are aligned by calendar month before constructing the model dataset. Only months available in both datasets are retained.

---

## Methodology

### 1. Data Preprocessing

Stages of data preprocessing for both datasets are described above.

### 2. EOF Analysis and Dimensionality Reduction

Empirical Orthogonal Function (EOF) analysis is used to identify dominant spatial patterns of variability in the climate datasets. The decomposition is performed using Singular Value Decomposition (SVD).

The analysis produces:

- **EOF patterns:** Spatial structures representing dominant modes of variability.
- **Principal Components (PCs):** Time series describing how strongly each spatial pattern is expressed over time.
- **Explained variance:** The proportion of total variability represented by each mode.

The spatial contribution of an EOF mode at a given time can be reconstructed by multiplying its spatial pattern by the corresponding principal component value.

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

The model uses six SST principal components and two seasonal features. The seasonal features (month_sin and month_cos) are derived from the target month during feature preparation.

The prediction target is the first principal component (PC1) of the MSWEP precipitation anomaly field.

### 4. Model Experiments

Following models were considered for predicting the target:

1. Baseline.
   Predicts zeros for every target value.
2. Linear regression.
   Grid search was used to select the regularization method (none, L1, L2, or Elastic Net) and its corresponding hyperparameters.
3. Random Forest.
   GridSearch was used to find the best parameters.
4. Gradient Boosting.
   GridSearch was used to find the best parameters.
5. Multi-Layer Perceptron.
   Simple model with two hidden layers and ReLU activation function.

MAE was selected as the primary metric for model selection and as a loss function because it measures the average absolute prediction error and is less sensitive to large errors than RMSE.

### 5. Model training

The final 60 months (2021–2025) were reserved as a held-out test set. Model selection and hyperparameter tuning were performed using time-series cross-validation on the training data. No separate validation set was created.

The following metrics are used: MAE, RMSE, R²

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline | 537.52 | 683.42 | -0.12 |
| LinReg | 515.07 | 652.75 | -0.03 |
| Random Forest | 507.19 | 621.43 | 0.071 |
| Gradient Boosting | 514.99 | 640.79 | 0.01 |
| MLP | 543.12 | 641.17 | 0.01 |

Random Forest achieved the best performance among the evaluated models, obtaining the lowest MAE (507.19) and RMSE (621.43), as well as the highest R² (0.071). Compared with the zero baseline, it reduced MAE by approximately 5.6% and RMSE by 9.1%. However, the relatively low R² indicates that the model's predictive skill on the held-out test period remains limited.

### 6. Chosen machine learning model

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

---

### Key Results
- Random Forest achieved the best performance among the five evaluated models on the held-out 2021–2025 test period.
- Compared with the zero baseline, Random Forest reduced MAE by approximately 5.6% and RMSE by 9.1%.
- The model achieved a positive R² of 0.071, suggesting that Indo-Pacific SST principal components contain some information relevant to predicting Brunei precipitation anomalies.
- However, the overall predictive performance remains modest, indicating that SST-based predictors alone may not be sufficient to accurately capture local precipitation variability.

---

## Notebooks

| Notebook | Description |
|---|---|
| `MSWEP_preprocessing.ipynb` | Preprocessing of precipitation data and EOF decomposition |
| `SST_preprocessing.ipynb` | SST preprocessing and EOF decomposition |
| `dataset_creation.ipynb` | Temporal alignment and construction of the model dataset |
| `training_and_evaluating_models.ipynb` | Model training, hyperparameter optimization, and evaluation |

The notebooks provide a reproducible overview of the analytical workflow.

---

## Application

The application is built with Streamlit and provides an interactive interface for generating and exploring precipitation predictions. The trained model generates predictions of the first principal component (PC1) of the precipitation anomaly field.

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
`target_month` specifies the month for which the precipitation anomaly is being predicted. No lags are used in this project.
The example values above are illustrative and do not represent actual observations.

The application derives `month_sin` and `month_cos` from `target_month`.

### Visualizations

The application includes:

- Spatial maps of predicted precipitation anomalies: reconstructed using the predicted precipitation PC1 and the corresponding first EOF spatial pattern.
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
│   ├── mswep_2024.nc, sst_2024.nc  # sample maps
│   ├── *_variance_truncation.csv   # explained variance
│   ├── *PC_train.nc, *PC_test.nc   # principal components
│   ├── sst_eof_svd_results.nc      # SST EOFs and PCs
│   ├── brunei_landmask_used_for_EOF_on_mswep_grid.nc
│   ├── world_countries.geojson
│   ├── mswep_brunei_landmask_EOF_patterns_10_fullgrid.nc
│   ├── brunei.geojson
│   └── malaysia.geojson
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── extra_functions/
│   ├── datasets_visualisation.py
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
git clone https://github.com/creveezo/brunei-rainfall-predictor.git

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
The relatively short temporal coverage of the datasets limits the amount of historical information available for model training.
- Performance on a single historical test period does not guarantee generalization to other climatic conditions.
- The SST features are from the same month as the target, so the project does not demonstrate an advance forecast. Raw SST must be converted to compatible PCs before upload.


---

## Future Improvements

Planned improvements include:

- Exploring alternative feature selection and dimensionality reduction approaches.
- Exploring additional climate predictors, including atmospheric variables.

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


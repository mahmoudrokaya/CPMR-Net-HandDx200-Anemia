# HCEF Structured Learning Framework

## Project Title

A Unified Optimization Framework for Tensorized Residual, Contrastive, and Ensemble Classification

## Description

This repository contains the complete implementation workflow for the Hybrid Contrastive–Ensemble Framework (HCE-F) and its temporal HCEF-LSTM extension. The code supports the experiments reported in the manuscript, including Jena Climate data auditing, daily aggregation, sliding-window generation, baseline forecasting, HCE-F model training, temporal HCEF-LSTM forecasting, ablation analysis, lookback-window sensitivity analysis, final comparative benchmarking, and external validation using biomedical benchmark datasets.

The workflow is designed to support full reproducibility of the experiments reported in the study.

## Repository Contents

The repository contains the following main scripts:

| Script                                                          | Purpose                                                                                                                                                                    |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Experiment_1_Jena_Data_Audit.py`                               | Audits the raw Jena Climate dataset, detects temporal structure, profiles variables, checks missing values, and generates exploratory tables and figures.                  |
| `Experiment_1B_Jena_Daily_Aggregation_and_Window_Generation.py` | Aggregates the Jena Climate data into daily summaries and generates sliding-window forecasting datasets using 3-, 7-, 14-, and 30-day lookback windows.                    |
| `Experiment_2_Jena_Baseline_Forecasting.py`                     | Trains conventional forecasting baselines including linear models, tree-based models, MLP, and persistence baseline.                                                       |
| `Experiment_3_Jena_HCEF_Model.py`                               | Implements the dense HCE-F forecasting model using residual representation learning and projection-based regression.                                                       |
| `Experiment_3B_Jena_HCEF_LSTM_Model.py`                         | Implements the sequence-aware Temporal HCEF-LSTM model with recurrent learning, projection head, residual gate, contrastive regularization, and ensemble prediction heads. |
| `Experiment_4_Jena_HCEF_Ablation_Study.py`                      | Performs ablation experiments comparing the full Temporal HCEF-LSTM model with variants excluding contrastive loss, residual gate, ensemble fusion, and other components.  |
| `Experiment_5_Jena_Lookback_Window_Sensitivity.py`              | Evaluates the effect of different lookback windows: 3, 7, 14, and 30 days.                                                                                                 |
| `Experiment_6_Jena_Final_Comparative_Summary.py`                | Produces the final comparative benchmarking summary across baselines, HCE-F, HCEF-LSTM, ablation variants, and lookback-window experiments.                                |
| `Experiment_7_HCEF_External_Validation_Breast_Cancer.py`        | Performs external validation on the Breast Cancer Wisconsin Diagnostic Dataset.                                                                                            |
| `Experiment_8_HCEF_External_Validation_Heart_Disease.py`        | Performs external validation on the UCI Heart Disease Dataset.                                                                                                             |
| `Folder_S.py`                                                   | Generates a folder-structure tree and file inventory for documentation and reproducibility.                                                                                |

## Dataset Information

This study uses three publicly available datasets.

### 1. Jena Climate Dataset

The primary forecasting experiments use the Jena Climate dataset, originally released by the Max Planck Institute for Biogeochemistry and publicly available through Kaggle.

Dataset DOI: https://doi.org/10.34740/KAGGLE/DSV/13825279

The Jena Climate dataset is used for:

* temporal data audit;
* daily aggregation;
* sliding-window generation;
* baseline forecasting;
* HCE-F forecasting;
* HCEF-LSTM forecasting;
* ablation analysis;
* lookback-window sensitivity analysis; and
* final comparative benchmarking.

### 2. Breast Cancer Wisconsin Diagnostic Dataset

External validation is performed using the Breast Cancer Wisconsin Diagnostic Dataset from the UCI Machine Learning Repository.

Dataset URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### 3. UCI Heart Disease Dataset

External validation is also performed using the UCI Heart Disease Dataset from the UCI Machine Learning Repository.

Dataset URL: https://archive.ics.uci.edu/dataset/45/heart+disease

No proprietary or restricted datasets are used.

## Input Data Requirements

The scripts assume the following local input folders:

```text
Data/
├── Baseline/
│   └── jena_climate_2009_2016_Excel.xlsx
├── Breast Cancer Wisconsin Diagnostic Dataset/
│   └── dataset file
└── UCI Heart Disease Data/
    └── dataset file
```

The original scripts use Windows absolute paths. Before running the workflow, update the `DATA_FILE`, `DATA_DIR`, and `RESULTS_ROOT` variables in each script to match the local directory structure.

## Output Structure

Each experiment creates its own output folder under the main results directory. Typical output subfolders include:

```text
Tables/
Figures/
Reports/
Predictions/
Models/
Processed_Data/
Generated_Datasets/
Model_Outputs/
```

The outputs include:

* CSV result tables;
* JSON summary reports;
* generated forecasting datasets;
* model predictions;
* trained model files;
* performance metrics;
* confusion matrices;
* ROC and precision–recall curves;
* residual plots;
* observed-vs-predicted figures;
* learning-curve figures; and
* final comparative benchmark summaries.

## Software Requirements

The code was developed using Python 3.x and requires the following packages:

```text
numpy
pandas
matplotlib
scikit-learn
torch
tensorflow
keras
openpyxl
```

Optional packages may be required depending on the local environment and data format.

## Installation

Create a new Python environment:

```bash
python -m venv hcef_env
```

Activate the environment.

On Windows:

```bash
hcef_env\Scripts\activate
```

On macOS/Linux:

```bash
source hcef_env/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn torch tensorflow keras openpyxl
```

Alternatively, create a `requirements.txt` file containing the package list above and install using:

```bash
pip install -r requirements.txt
```

## Recommended Execution Order

Run the scripts in the following order:

```bash
python Experiment_1_Jena_Data_Audit.py
python Experiment_1B_Jena_Daily_Aggregation_and_Window_Generation.py
python Experiment_2_Jena_Baseline_Forecasting.py
python Experiment_3_Jena_HCEF_Model.py
python Experiment_3B_Jena_HCEF_LSTM_Model.py
python Experiment_4_Jena_HCEF_Ablation_Study.py
python Experiment_5_Jena_Lookback_Window_Sensitivity.py
python Experiment_6_Jena_Final_Comparative_Summary.py
python Experiment_7_HCEF_External_Validation_Breast_Cancer.py
python Experiment_8_HCEF_External_Validation_Heart_Disease.py
python Folder_S.py
```

## Experimental Workflow

### Experiment 1: Jena Climate Data Audit

This script loads the raw Jena Climate dataset, detects the datetime column, sorts the time series, profiles numeric variables, checks missing values and duplicate timestamps, and generates audit reports.

Main outputs include:

* dataset summary table;
* column profile table;
* temporal structure report;
* exploratory figures;
* processed data files.

### Experiment 1B: Daily Aggregation and Sliding-Window Generation

This script converts the high-frequency Jena Climate data into daily aggregated features. For each numeric variable, daily mean, minimum, maximum, and standard deviation are computed. Sliding-window datasets are then generated for 3-, 7-, 14-, and 30-day lookback windows.

Main outputs include:

* `jena_daily_aggregated.csv`;
* `jena_daily_window_3d.csv`;
* `jena_daily_window_7d.csv`;
* `jena_daily_window_14d.csv`;
* `jena_daily_window_30d.csv`;
* summary tables.

### Experiment 2: Baseline Forecasting

This script evaluates conventional forecasting baselines on the 7-day Jena sliding-window dataset. The evaluated models include:

* Linear Regression;
* Ridge Regression;
* ElasticNet;
* Random Forest;
* Extra Trees;
* Gradient Boosting;
* MLP Regressor;
* persistence baseline.

The data are split chronologically using 70% training, 15% validation, and 15% testing.

Evaluation metrics include:

* MAE;
* RMSE;
* MAPE;
* SMAPE;
* R².

### Experiment 3: Dense HCE-F Forecasting Model

This script implements the dense HCE-F forecasting model using PyTorch. The model includes:

* feature scaling;
* dense encoder;
* projection head;
* regression head;
* chronological train/validation/test split;
* model training and evaluation.

### Experiment 3B: Temporal HCEF-LSTM Model

This script implements the sequence-aware Temporal HCEF-LSTM model. The model includes:

* input projection;
* LSTM sequence encoder;
* residual gate;
* projection head;
* two regression heads;
* softmax-based ensemble weighting;
* contrastive regularization;
* early stopping.

The default configuration includes:

* 7-day lookback window;
* batch size 32;
* learning rate 5 × 10⁻⁴;
* weight decay 1 × 10⁻⁴;
* LSTM hidden dimension 96;
* 2 LSTM layers;
* projection dimension 64;
* dropout 0.25;
* contrastive weight 0.02.

### Experiment 4: Ablation Study

This script evaluates the contribution of the major architectural components of the Temporal HCEF-LSTM model.

The tested variants are:

1. Full Temporal HCEF-LSTM;
2. No Contrastive Loss;
3. No Residual Gate;
4. Single Head without Ensemble;
5. LSTM Only.

The purpose is to quantify the contribution of contrastive regularization, residual gating, projection learning, and ensemble prediction.

### Experiment 5: Lookback-Window Sensitivity Analysis

This script evaluates the full Temporal HCEF-LSTM model using different historical windows:

* 3 days;
* 7 days;
* 14 days;
* 30 days.

This experiment examines how the amount of temporal context affects forecasting accuracy.

### Experiment 6: Final Comparative Summary

This script combines the results from the baseline forecasting, HCE-F, HCEF-LSTM, ablation, and lookback-window experiments into a final comparative benchmark.

The output includes final comparison tables and summary figures.

### Experiment 7: External Validation on Breast Cancer Wisconsin Diagnostic Dataset

This script evaluates the HCE-F framework on an independent biomedical classification dataset. It automatically detects and preprocesses the dataset, encodes labels, scales features, trains the model, and evaluates performance.

Evaluation metrics include:

* accuracy;
* precision;
* recall;
* F1-score;
* ROC-AUC;
* PR-AUC;
* confusion matrix.

### Experiment 8: External Validation on UCI Heart Disease Dataset

This script evaluates the HCE-F framework on the UCI Heart Disease Dataset. The target is converted into binary disease status where appropriate.

Evaluation metrics include:

* accuracy;
* precision;
* recall;
* F1-score;
* ROC-AUC;
* PR-AUC;
* confusion matrix.

## Reproducibility Settings

The experiments use fixed random seeds where applicable:

```python
SEED = 42
```

The scripts set seeds for:

* Python `random`;
* NumPy;
* PyTorch;
* CUDA, when available.

Chronological splitting is used for the Jena forecasting experiments to avoid temporal leakage. Stratified splitting is used for external classification experiments where appropriate.

## Computational Environment

The experiments were designed to run on a standard research workstation. GPU acceleration is used automatically when available through PyTorch:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

The workflow can also run on CPU, although training time may increase for the HCEF-LSTM and external validation experiments.

## Evaluation Metrics

For forecasting experiments, the following metrics are reported:

* Mean Absolute Error (MAE);
* Root Mean Squared Error (RMSE);
* Mean Absolute Percentage Error (MAPE);
* Symmetric Mean Absolute Percentage Error (SMAPE);
* coefficient of determination (R²).

For classification experiments, the following metrics are reported:

* accuracy;
* precision;
* recall;
* F1-score;
* ROC-AUC;
* PR-AUC.

## Methodological Summary

The HCE-F framework integrates three main components:

1. Residual feature encoding to preserve stable information flow and model nonlinear feature interactions.
2. Contrastive representation learning to improve latent-space structure and reduce representation noise.
3. Ensemble prediction fusion to stabilize decision-level output and reduce predictive variance.

The Temporal HCEF-LSTM extension adds recurrent sequence modeling to better capture temporal dependencies in the Jena Climate forecasting task.

## Notes for Reuse

Before running the scripts on another system:

1. Download the required datasets from the original public sources.
2. Place the datasets in the expected local folders.
3. Update all path variables in the scripts.
4. Install the required Python packages.
5. Run the scripts in the recommended order.
6. Verify that each experiment produces the expected output folders and result files.

## Citation

If this code or workflow is used, please cite the associated manuscript:

Rokaya M, Hemdan DI, Gad I, Baghdadi NA, Malki A, Atlam E. A Unified Optimization Framework for Tensorized Residual, Contrastive, and Ensemble Classification.

Dataset citations:

* Jena Climate Dataset: https://doi.org/10.34740/KAGGLE/DSV/13825279
* Breast Cancer Wisconsin Diagnostic Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
* UCI Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

## License

Please add the final repository license here before submission. Recommended open-source licenses include:

* MIT License;
* Apache License 2.0;
* BSD 3-Clause License.

If no reuse license is intended, state that all rights are reserved and contact the corresponding author for permission.

## Contact

For questions about the implementation or reproducibility workflow, please contact the corresponding author listed in the manuscript.

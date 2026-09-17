# Charlotte Neighborhood Housing Price Analysis

A data science project analyzing how economic, educational, housing, and public-safety factors relate to neighborhood housing prices in Charlotte, North Carolina.

## Project Overview

This project uses neighborhood-level data from Charlotte to investigate factors associated with housing prices and apply machine learning models to housing-related prediction problems.

The project has two primary objectives:

1. **Regression:** Predict 2023 neighborhood housing prices.
2. **Classification:** Classify neighborhoods as having 2023 housing prices above or below the dataset median.

Multiple datasets were cleaned and combined using Charlotte Neighborhood Profile Area (NPA) identifiers to create a unified dataset for analysis and machine learning.

The final combined dataset contains **459 neighborhoods**, with **409 complete observations** available for modeling after removing rows with missing values.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- SQL / SQLite
- Matplotlib
- Git / GitHub

## Data Preparation

The project combines five neighborhood-level source datasets:

- Employment
- High School Graduation Rate
- Home Sales Price
- Job Density
- CMPD Homicide Data

The data-preparation pipeline uses Python, pandas, and SQLite to:

- Clean percentage and currency values
- Convert relevant variables to numeric formats
- Load source datasets into an SQLite database
- Aggregate homicide incidents by NPA
- Join datasets using NPA identifiers
- Create a unified dataset for machine learning
- Export the processed dataset as `final_fixed_dataset.csv`

## Features Used

The machine learning models use the following predictors:

- 2023 employment rate
- 2023 high school graduation rate
- 2022 job density
- Homicide count
- 2021 home price

For regression, the target variable is **2023 home price**.

For classification, 2023 home prices are converted into a binary target based on the dataset median:

- `1` = Home price above the median
- `0` = Home price at or below the median

## Regression Analysis

Three regression algorithms were evaluated:

- Linear Regression
- K-Nearest Neighbors (KNN) Regression
- Random Forest Regression

KNN hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation.

Models were evaluated using:

- Root Mean Squared Error (RMSE)
- R²
- Cross-validation

The regression analysis also includes Random Forest feature importance, actual-vs-predicted visualization, residual analysis, and model RMSE comparison.

### Regression Results

| Model | Test RMSE | Test R² |
| --- | ---: | ---: |
| Linear Regression | $280,805 | 0.395 |
| K-Nearest Neighbors | $327,007 | 0.180 |
| Random Forest | $300,969 | 0.310 |

Among the evaluated regression models, **Linear Regression produced the lowest held-out RMSE and highest held-out R²**.

Its R² of approximately **0.395** indicates that the model explained about 39.5% of the observed variation in 2023 neighborhood housing prices in the test data.

Random Forest feature-importance analysis identified **2021 home price** as the strongest predictor. Employment and high school graduation rates were also important, while job density and homicide count contributed less.

## Classification Analysis

Three classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

KNN hyperparameter tuning was performed using `GridSearchCV` with ROC-AUC as the scoring metric.

Models were evaluated using:

- Accuracy
- ROC-AUC
- Confusion matrices
- Classification reports
- 5-fold cross-validation
- ROC curves

### Classification Results

| Model | Test Accuracy | Test ROC-AUC |
| --- | ---: | ---: |
| Logistic Regression | 62.2% | 0.651 |
| K-Nearest Neighbors (k=5) | 65.9% | 0.685 |
| Random Forest | 63.4% | 0.662 |

Among the evaluated classification models, **K-Nearest Neighbors achieved the highest held-out ROC-AUC**, approximately **0.685**, along with an accuracy of approximately **65.9%**.

Across the regression and classification analyses, previous housing prices were the most informative predictor of 2023 neighborhood housing-price outcomes.

## Repository Structure

```text
charlotte-housing-analysis/
│
├── data/
│   ├── CMPD_Homicide.csv
│   ├── Employment.csv
│   ├── High School Graduation Rate.csv
│   ├── Home Sales Price.csv
│   └── Job Density.csv
│
├── ClassificationModel.py
├── FinalFixedDatasetSQL.py
├── RegressionModel.py
├── final_fixed_dataset.csv
└── README.md

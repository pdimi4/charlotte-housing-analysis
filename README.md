# Charlotte Neighborhood Housing Price Analysis

A data science project analyzing how economic, educational, housing, and public-safety factors relate to neighborhood housing prices in Charlotte, North Carolina.

## Project Overview

This project analyzes neighborhood-level data in Charlotte to explore factors associated with housing prices and apply machine learning models to housing-related prediction problems.

The project has two primary objectives:

1. **Regression:** Predict 2023 neighborhood housing prices.
2. **Classification:** Classify neighborhoods as having 2023 housing prices above or below the dataset median.

Data from multiple sources was combined using Charlotte Neighborhood Profile Area (NPA) identifiers to create a unified dataset for analysis and modeling.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- SQL / SQLite
- Matplotlib
- Git / GitHub

## Data Preparation

Multiple neighborhood-level datasets were cleaned and combined using Python, pandas, and SQLite.

The data pipeline:

- Cleans percentage and currency values
- Converts relevant fields into numeric formats
- Loads datasets into an SQLite database
- Aggregates homicide incidents by NPA
- Joins datasets using NPA identifiers
- Creates a combined dataset for machine learning

### Features Used

The modeling scripts use the following features:

- 2023 employment rate
- 2023 high school graduation rate
- 2022 job density
- Homicide count
- 2021 home price

The regression target is the neighborhood's 2023 home price.

## Regression Analysis

The regression portion of the project predicts 2023 neighborhood housing prices.

### Models

- Linear Regression
- K-Nearest Neighbors (KNN) Regression
- Random Forest Regression

KNN hyperparameter tuning is performed using `GridSearchCV` with 5-fold cross-validation.

### Evaluation

Regression models are evaluated using:

- Root Mean Squared Error (RMSE)
- R²
- 5-fold cross-validation

The regression analysis also includes:

- Random Forest feature importance
- Actual vs. predicted visualization
- Residual analysis
- Model RMSE comparison

## Classification Analysis

The classification portion converts 2023 housing prices into a binary target based on the dataset median.

`high_price_2023`

- `1` = Home price above the median
- `0` = Home price at or below the median

### Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

KNN hyperparameter tuning is performed using `GridSearchCV` with ROC-AUC as the scoring metric.

### Evaluation

Classification models are evaluated using:

- Accuracy
- ROC-AUC
- Confusion matrices
- Classification reports
- 5-fold cross-validation

The classification analysis also compares ROC curves and examines Random Forest feature importance.

## Repository Structure

| File | Description |
| --- | --- |
| `FinalFixedDatasetSQL.py` | Cleans and combines neighborhood datasets using pandas and SQLite |
| `RegressionModel.py` | Trains and evaluates regression models for 2023 housing-price prediction |
| `ClassificationModel.py` | Trains and evaluates classification models for above/below-median housing prices |

## Project Background

This project was originally completed as a group final project for **DTSC 2302: Modeling and Society** at the **University of North Carolina at Charlotte**.

### Project Team

- Pavle Dimitrijevic
- Ikumi Uemura
- Michael Forshay
- Sonia Sun
- James Harris

This repository is a portfolio-oriented version of the project maintained by Pavle Dimitrijevic. The original academic project was collaborative, and credit for the group work belongs to all team members listed above.

## Skills Demonstrated

- Python programming
- Data cleaning and preprocessing
- SQL-based data integration
- pandas and NumPy
- Regression modeling
- Classification modeling
- Machine learning model evaluation
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Data visualization
- Working with real-world neighborhood data
- Collaborative data science development

## Future Improvements

This portfolio version is being improved to make the analysis more reproducible and easier to run outside the original development environment.

Planned improvements include:

- Replacing local file paths with a portable project structure
- Saving the processed dataset directly from the data-preparation pipeline
- Adding final model-performance results
- Adding selected model visualizations
- Adding dependency documentation

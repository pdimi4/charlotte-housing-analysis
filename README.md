# Charlotte Neighborhood Housing Price Analysis

A data science project analyzing how economic, educational, housing, and public-safety factors relate to neighborhood housing prices in Charlotte, North Carolina.

## Project Overview

This project uses neighborhood-level data from Charlotte to investigate factors associated with housing prices and to build predictive models for neighborhood housing values.

The analysis has two primary objectives:

1. **Regression:** Predict 2023 neighborhood housing prices.
2. **Classification:** Classify neighborhoods as above or below the median 2023 housing price.

Data from multiple sources was combined at the Charlotte Neighborhood Profile Area (NPA) level to create a unified dataset for analysis and modeling.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- SQL / SQLite
- Matplotlib
- Seaborn
- Git / GitHub

## Data Sources

The project incorporates neighborhood-level information from sources including:

- Charlotte-Mecklenburg Quality of Life data
- City of Charlotte Open Data
- Charlotte-Mecklenburg Police Department (CMPD) homicide records

### Variables Analyzed

Key variables include:

- Employment rate
- High school graduation rate
- Job density
- Home sales prices
- Homicide count
- Previous-year housing prices

Datasets were merged using Charlotte's Neighborhood Profile Area (NPA) identifiers. Homicide incidents were aggregated to the neighborhood level before being incorporated into the modeling dataset.

## Regression Analysis

The regression portion of the project predicts `home_price_2023`.

### Features

- `employment_2023`
- `grad_2023`
- `job_density_2022`
- `homicide_count`
- `home_price_2021`

### Models

- Linear Regression
- K-Nearest Neighbors Regression
- Decision Tree Regression
- Random Forest Regression

### Evaluation

Models are evaluated using:

- Root Mean Squared Error (RMSE)
- R²
- Cross-validation

The analysis also examines feature importance to better understand which neighborhood characteristics are most strongly associated with housing prices.

## Classification Analysis

The classification portion predicts whether a neighborhood's 2023 housing price is above or below the citywide median.

### Target

`high_price_2023`

- `1` = Above median housing price
- `0` = Below median housing price

### Models

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

### Evaluation

Classification performance is evaluated using:

- Accuracy
- ROC-AUC
- Confusion matrices
- Cross-validation

## Repository Structure

| File | Description |
| --- | --- |
| `FinalFixedDatasetSQL.py` | Data preparation, SQL-based dataset merging, and feature construction |
| `RegressionModel.py` | Regression modeling and evaluation |
| `ClassificationModel.py` | Classification modeling, tuning, and evaluation |

## Project Background

This analysis was originally completed as a group final project for **DTSC 2302: Modeling and Society** at the **University of North Carolina at Charlotte**.

### Project Team

- Pavle Dimitrijevic
- Ikumi Uemura
- Michael Forshay
- Sonia Sun
- James Harris

This repository is a portfolio-oriented version of the project maintained by Pavle Dimitrijevic. The original project was collaborative, and credit for the group work belongs to all team members listed above.

## Skills Demonstrated

- Data cleaning and integration
- SQL-based data preparation
- Exploratory data analysis
- Regression modeling
- Classification modeling
- Model evaluation
- Cross-validation and model tuning
- Working with real-world public datasets
- Collaborative data science workflow

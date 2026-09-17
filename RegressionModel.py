# RegressionModel.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv("final_fixed_dataset.csv")
df = df.dropna()

# -------------------------
# 2. Feature Selection
# -------------------------
features = [
    "employment_2023",
    "grad_2023",
    "job_density_2022",
    "homicide_count",
    "home_price_2021"
]

X = df[features]
y = df["home_price_2023"]

# -------------------------
# 3. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. Scaling (for Linear + KNN)
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 5. KNN Tuning (CV)
# -------------------------
param_grid = {
    "n_neighbors": list(range(3, 16, 2))
}

knn_grid = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

knn_grid.fit(X_train_scaled, y_train)

best_k = knn_grid.best_params_["n_neighbors"]
print(f"\nBest K for KNN (CV): {best_k}")

# -------------------------
# 6. Define Models
# -------------------------
models = {
    "Linear Regression": LinearRegression(),
    f"KNN (k={best_k})": KNeighborsRegressor(n_neighbors=best_k),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# -------------------------
# 7. Train Models
# -------------------------
trained_models = {}

trained_models["Linear Regression"] = models["Linear Regression"].fit(X_train_scaled, y_train)
trained_models[f"KNN (k={best_k})"] = models[f"KNN (k={best_k})"].fit(X_train_scaled, y_train)
trained_models["Random Forest"] = models["Random Forest"].fit(X_train, y_train)

# -------------------------
# 8. Evaluation Function
# -------------------------
results = []

def evaluate_model(name, model, X_te):
    y_pred = model.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    results.append((name, rmse, r2))

    print(f"\n{name}")
    print("-" * 40)
    print("RMSE:", round(rmse, 4))
    print("R²:", round(r2, 4))

# Run evaluations
evaluate_model("Linear Regression", trained_models["Linear Regression"], X_test_scaled)
evaluate_model(f"KNN (k={best_k})", trained_models[f"KNN (k={best_k})"], X_test_scaled)
evaluate_model("Random Forest", trained_models["Random Forest"], X_test)

# -------------------------
# 9. Model Comparison
# -------------------------
print("\nModel Comparison (RMSE, R²):")
for r in results:
    print(r)

best = min(results, key=lambda x: x[1])  # lowest RMSE
print(f"\nBest Model: {best[0]} with RMSE = {round(best[1], 4)}")

# -------------------------
# 10. Cross Validation
# -------------------------
print("\nCross-Validation (RMSE):")

for name, model in models.items():
    if "Random Forest" in name:
        X_use = X
    else:
        X_use = scaler.fit_transform(X)

    scores = cross_val_score(model, X_use, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    print(f"{name}: Mean RMSE = {round(rmse_scores.mean(), 4)}")

# -------------------------
# 11. Feature Importance
# -------------------------
rf = trained_models["Random Forest"]

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances (Random Forest):")
print(feature_importance)

# -------------------------
# 12. Actual vs Predicted
# -------------------------
plt.figure()

y_pred_rf = trained_models["Random Forest"].predict(X_test)

plt.scatter(y_test, y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Random Forest)")

plt.show()

# -------------------------
# 13. Residual Plot
# -------------------------
residuals = y_test - y_pred_rf

plt.figure()
plt.scatter(y_pred_rf, residuals)

plt.axhline(0, linestyle='--')

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Random Forest)")

plt.show()

# -------------------------
# 14. Feature Importance Plot
# -------------------------
plt.figure()

plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")

plt.show()

# -------------------------
# 15. RMSE Comparison Plot
# -------------------------
names = [r[0] for r in results]
rmse_vals = [r[1] for r in results]

plt.figure()
plt.bar(names, rmse_vals)

plt.ylabel("RMSE")
plt.title("Model RMSE Comparison")

plt.show()

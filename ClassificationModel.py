# ClassificationModel.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv("final_fixed_dataset.csv")
df = df.dropna()

# -------------------------
# 2. Create Target Variable
# -------------------------
df["high_price_2023"] = (df["home_price_2023"] > df["home_price_2023"].median()).astype(int)

# -------------------------
# 3. Feature Selection
# -------------------------
features = [
    "employment_2023",
    "grad_2023",
    "job_density_2022",
    "homicide_count",
    "home_price_2021"
]

X = df[features]
y = df["high_price_2023"]

# -------------------------
# 4. Class Balance
# -------------------------
print("\nClass Distribution:")
print(y.value_counts())

print("\nClass Proportions:")
print(y.value_counts(normalize=True))

# -------------------------
# 5. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 6. Scaling
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# 7. KNN Tuning (Proper CV)
# -------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_neighbors": list(range(3, 16, 2))
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc'
)

knn_grid.fit(X_train_scaled, y_train)

best_k = knn_grid.best_params_["n_neighbors"]
print(f"\nBest K for KNN (CV): {best_k}")

# -------------------------
# 8. Define Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    f"KNN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# -------------------------
# 9. Train Models
# -------------------------
trained_models = {}

trained_models["Logistic Regression"] = models["Logistic Regression"].fit(X_train_scaled, y_train)
trained_models[f"KNN (k={best_k})"] = models[f"KNN (k={best_k})"].fit(X_train_scaled, y_train)
trained_models["Random Forest"] = models["Random Forest"].fit(X_train, y_train)

# -------------------------
# 10. Evaluation
# -------------------------
results = []

def evaluate_model(name, model, X_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    results.append((name, acc, auc))

    print(f"\n{name}")
    print("-" * 40)
    print("Accuracy:", round(acc, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Run evaluations
evaluate_model("Logistic Regression", trained_models["Logistic Regression"], X_test_scaled)
evaluate_model(f"KNN (k={best_k})", trained_models[f"KNN (k={best_k})"], X_test_scaled)
evaluate_model("Random Forest", trained_models["Random Forest"], X_test)

# -------------------------
# 11. Model Comparison
# -------------------------
print("\nModel Comparison (Accuracy, ROC-AUC):")
for r in results:
    print(r)

best = max(results, key=lambda x: x[2])
print(f"\nBest Model: {best[0]} with ROC-AUC = {round(best[2], 4)}")

# -------------------------
# 12. Cross Validation
# -------------------------
print("\nCross-Validation (ROC-AUC):")

for name, model in models.items():
    if "Random Forest" in name:
        X_use = X
    else:
        X_use = scaler.fit_transform(X)

    scores = cross_val_score(model, X_use, y, cv=5, scoring='roc_auc')
    print(f"{name}: Mean ROC-AUC = {round(scores.mean(), 4)}")

# -------------------------
# 13. Feature Importance
# -------------------------
rf = trained_models["Random Forest"]

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances (Random Forest):")
print(feature_importance)

# -------------------------
# 14. ROC Curve
# -------------------------
plt.figure()

for name, model in trained_models.items():
    if "Random Forest" in name:
        X_te = X_test
    else:
        X_te = X_test_scaled

    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()
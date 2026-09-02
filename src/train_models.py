"""
Train and evaluate four classifiers on the acquisition-target dataset:
  - Logistic Regression (interpretable baseline)
  - Decision Tree
  - SVM (RBF kernel)
  - Random Forest (ensemble comparison)

With only 36 labeled companies (16 targets / 20 controls), a train/test split
would leave test folds with a handful of companies -- too noisy to trust.
Instead we use Leave-One-Out Cross-Validation (LOOCV): each company is held
out once, the model is trained on the other 35, and scored on the one left
out. This uses every data point for both training and testing without ever
leaking a company's own label into its own training fold, and is the standard
approach for small-N classification problems.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    from .prepare_data import build_dataset, FEATURE_COLS
except ImportError:
    from prepare_data import build_dataset, FEATURE_COLS

RANDOM_STATE = 8

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=RANDOM_STATE),
    "svm_rbf": CalibratedClassifierCV(SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), ensemble=False),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=RANDOM_STATE),
}


def loocv_evaluate(model, X, y):
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        m = clone(model)
        m.fit(X_train_s, y_train)

        pred = m.predict(X_test_s)[0]
        if hasattr(m, "decision_function"):
            prob = m.decision_function(X_test_s)[0]   # deterministic ranking score
        elif hasattr(m, "predict_proba"):
            prob = m.predict_proba(X_test_s)[0][1]
        else:
            prob = pred

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_prob.append(prob)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def main():
    df, _ = build_dataset()
    X = df[FEATURE_COLS].values
    y = df["label"].values
    companies = df["company"].values

    results = {}
    predictions = {}

    for name, model in MODELS.items():
        y_true, y_pred, y_prob = loocv_evaluate(model, X, y)
        results[name] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 3),
        }
        predictions[name] = y_prob.tolist()
        print(f"{name:22s} | acc={results[name]['accuracy']:.3f}  "
              f"prec={results[name]['precision']:.3f}  rec={results[name]['recall']:.3f}  "
              f"f1={results[name]['f1']:.3f}  roc_auc={results[name]['roc_auc']:.3f}")

    with open("outputs/model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-company LOOCV predicted probabilities for the best model (by ROC-AUC)
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest model by LOOCV ROC-AUC: {best_model_name}")

    pred_df = pd.DataFrame({
        "company": companies,
        "actual_label": y,
        f"{best_model_name}_predicted_prob": predictions[best_model_name],
    }).sort_values(f"{best_model_name}_predicted_prob", ascending=False)
    pred_df.to_csv("outputs/loocv_predictions.csv", index=False)

    # Fit the best model on ALL data (for the live screener app) + feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_model = clone(MODELS[best_model_name])
    final_model.fit(X_scaled, y)

    import joblib
    joblib.dump({"model": final_model, "scaler": scaler, "features": FEATURE_COLS,
                 "model_name": best_model_name}, "outputs/final_model.joblib")

    if hasattr(final_model, "coef_"):
        importance = dict(zip(FEATURE_COLS, final_model.coef_[0].round(3).tolist()))
    elif hasattr(final_model, "feature_importances_"):
        importance = dict(zip(FEATURE_COLS, final_model.feature_importances_.round(3).tolist()))
    elif hasattr(final_model, "base_estimator_"):
        # Handle CalibratedClassifierCV and similar wrappers
        base = final_model.base_estimator_
        if hasattr(base, "coef_"):
            importance = dict(zip(FEATURE_COLS, base.coef_[0].round(3).tolist()))
        elif hasattr(base, "feature_importances_"):
            importance = dict(zip(FEATURE_COLS, base.feature_importances_.round(3).tolist()))
        else:
            importance = {}
    else:
        importance = {}
    with open("outputs/feature_importance.json", "w") as f:
        json.dump({"model": best_model_name, "importance": importance}, f, indent=2)
    print("\nFeature importance / coefficients:")
    for k, v in sorted(importance.items(), key=lambda x: -abs(x[1])):
        print(f"  {k:28s} {v:+.3f}")


if __name__ == "__main__":
    main()

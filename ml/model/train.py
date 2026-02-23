import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic churn prediction dataset.
    Replace this with your actual data loading logic.
    """
    np.random.seed(42)

    df = pd.DataFrame({
        "tenure_months":       np.random.randint(1, 72, n_samples),
        "monthly_charges":     np.random.uniform(20, 120, n_samples),
        "total_charges":       np.random.uniform(100, 8000, n_samples),
        "num_products":        np.random.randint(1, 6, n_samples),
        "support_calls":       np.random.randint(0, 10, n_samples),
        "payment_delay_days":  np.random.randint(0, 30, n_samples),
        "contract_length":     np.random.choice([1, 12, 24], n_samples),
        "has_online_backup":   np.random.randint(0, 2, n_samples),
        "has_tech_support":    np.random.randint(0, 2, n_samples),
    })

    # Churn probability influenced by features
    churn_prob = (
        0.3 * (df["tenure_months"] < 12).astype(int) +
        0.2 * (df["support_calls"] > 5).astype(int) +
        0.2 * (df["payment_delay_days"] > 15).astype(int) +
        0.1 * (df["contract_length"] == 1).astype(int) +
        0.2 * np.random.random(n_samples)
    )
    df["churned"] = (churn_prob > 0.5).astype(int)

    return df


def train(
    experiment_name: str = "churn-prediction",
    model_output_path: str = "ml/model/model.joblib",
    metrics_output_path: str = "ml/model/metrics.json",
) -> dict:
    """
    Train model, track with MLflow, save artifacts.
    Returns metrics dict for validation gate.
    """

    # ── MLflow Setup ──────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="gradient-boosting-v1") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # ── Data ──────────────────────────────────────────────
        logger.info("Loading data...")
        df = generate_sample_data()

        FEATURES = [
            "tenure_months", "monthly_charges", "total_charges",
            "num_products", "support_calls", "payment_delay_days",
            "contract_length", "has_online_backup", "has_tech_support"
        ]
        TARGET = "churned"

        X = df[FEATURES]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # ── Model ─────────────────────────────────────────────
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
        }

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(**params))
        ])

        # ── Training ──────────────────────────────────────────
        logger.info("Training model...")
        model.fit(X_train, y_train)

        # ── Cross Validation ──────────────────────────────────
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Evaluation ────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":      round(accuracy_score(y_test, y_pred), 4),
            "precision":     round(precision_score(y_test, y_pred), 4),
            "recall":        round(recall_score(y_test, y_pred), 4),
            "f1_score":      round(f1_score(y_test, y_pred), 4),
            "roc_auc":       round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc":    round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
            "features":      FEATURES,
            "model_type":    "GradientBoostingClassifier",
            "run_id":        run.info.run_id,
        }

        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")

        # ── MLflow Logging ────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items()
                           if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="churn-prediction-model"
        )

        # ── Save Artifacts ────────────────────────────────────
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

        joblib.dump(model, model_output_path)
        logger.info(f"Model saved → {model_output_path}")

        with open(metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved → {metrics_output_path}")

        return metrics


if __name__ == "__main__":
    metrics = train()
    print(f"\n✅ Training complete. ROC-AUC: {metrics['roc_auc']}")
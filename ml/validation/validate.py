import json
import time
import logging
import sys
import os
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"


# ── Thresholds ─────────────────────────────────────────────────────────────
# Adjust these per your model and business requirements
THRESHOLDS = {
    "accuracy":         0.80,   # Min accuracy
    "roc_auc":          0.82,   # Min ROC-AUC — most important for churn
    "f1_score":         0.75,   # Min F1 score
    "precision":        0.70,   # Min precision
    "recall":           0.70,   # Min recall
    "cv_roc_auc":       0.80,   # Min cross-val ROC-AUC (generalization check)
    "max_model_mb":     100,    # Max model file size in MB
    "max_inference_ms": 100,    # Max inference time per batch (ms)
}

REQUIRED_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls", "payment_delay_days",
    "contract_length", "has_online_backup", "has_tech_support"
]


def check_performance_thresholds(metrics: dict) -> tuple[bool, list]:
    """Check all metrics meet minimum thresholds."""
    failures = []
    checks = ["accuracy", "roc_auc", "f1_score", "precision", "recall", "cv_roc_auc"]

    for metric in checks:
        if metric not in metrics:
            failures.append(f"Missing metric: {metric}")
            continue

        value = metrics[metric]
        threshold = THRESHOLDS[metric]

        if value < threshold:
            failures.append(
                f"{metric}: {value:.4f} < threshold {threshold:.4f}"
            )
        else:
            logger.info(f"{PASS} {metric}: {value:.4f} >= {threshold:.4f}")

    return len(failures) == 0, failures


def check_model_size(model_path: str) -> tuple[bool, str]:
    """Model file shouldn't be unreasonably large."""
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    max_mb = THRESHOLDS["max_model_mb"]

    if size_mb > max_mb:
        return False, f"Model size {size_mb:.1f}MB exceeds {max_mb}MB limit"

    logger.info(f"{PASS} Model size: {size_mb:.2f}MB (limit: {max_mb}MB)")
    return True, ""


def check_inference_speed(model_path: str) -> tuple[bool, str]:
    """Benchmark inference speed on a sample batch."""
    model = joblib.load(model_path)
    max_ms = THRESHOLDS["max_inference_ms"]

    # Generate a realistic sample batch (100 records)
    np.random.seed(42)
    sample = np.random.rand(100, len(REQUIRED_FEATURES))

    # Warm up
    model.predict(sample[:5])

    # Benchmark
    start = time.perf_counter()
    model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if elapsed_ms > max_ms:
        return False, f"Inference too slow: {elapsed_ms:.1f}ms > {max_ms}ms limit"

    logger.info(f"{PASS} Inference speed: {elapsed_ms:.1f}ms for 100 samples (limit: {max_ms}ms)")
    return True, ""


def check_schema(metrics: dict) -> tuple[bool, str]:
    """Validate that model was trained on expected features."""
    trained_features = set(metrics.get("features", []))
    expected_features = set(REQUIRED_FEATURES)

    missing = expected_features - trained_features
    extra = trained_features - expected_features

    if missing:
        return False, f"Missing required features: {missing}"
    if extra:
        logger.warning(f"⚠️  Extra features in model (not in schema): {extra}")

    logger.info(f"{PASS} Feature schema: all {len(expected_features)} required features present")
    return True, ""


def check_prediction_sanity(model_path: str) -> tuple[bool, str]:
    """
    Model shouldn't predict the same class for everything.
    A model that always predicts 0 or always 1 is broken.
    Uses realistic feature ranges — not random noise.
    """
    model = joblib.load(model_path)
    np.random.seed(123)

    n = 200
    # Generate samples using REALISTIC feature ranges
    # matching the actual training data distribution
    sample = np.column_stack([
        np.random.randint(1, 72, n),          # tenure_months
        np.random.uniform(20, 120, n),         # monthly_charges
        np.random.uniform(100, 8000, n),       # total_charges
        np.random.randint(1, 6, n),            # num_products
        np.random.randint(0, 10, n),           # support_calls
        np.random.randint(0, 30, n),           # payment_delay_days
        np.random.choice([1, 12, 24], n),      # contract_length
        np.random.randint(0, 2, n),            # has_online_backup
        np.random.randint(0, 2, n),            # has_tech_support
    ])
    predictions = model.predict(sample)

    unique_classes = len(set(predictions))
    if unique_classes < 2:
        return False, f"Model only predicts class {predictions[0]} — degenerate model!"

    pred_rate = predictions.mean()
    if pred_rate < 0.05 or pred_rate > 0.95:
        return False, f"Extreme prediction rate: {pred_rate:.2%} — model may be degenerate"

    logger.info(f"{PASS} Prediction sanity: {unique_classes} classes, {pred_rate:.2%} positive rate")
    return True, ""


def run_validation(
    metrics_path: str = "ml/model/metrics.json",
    model_path: str = "ml/model/model.joblib",
) -> bool:
    """
    Run all validation checks.
    Returns True if model passes ALL checks, False if ANY check fails.
    """
    logger.info("\n" + "="*60)
    logger.info("🔍 RUNNING MODEL VALIDATION GATE")
    logger.info("="*60)

    # Load metrics
    if not os.path.exists(metrics_path):
        logger.error(f"{FAIL} Metrics file not found: {metrics_path}")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)

    logger.info(f"Model: {metrics.get('model_type', 'unknown')}")
    logger.info(f"Run ID: {metrics.get('run_id', 'unknown')}\n")

    failures = []

    # ── Run all checks ─────────────────────────────────────────
    logger.info("--- 1. Performance Thresholds ---")
    passed, perf_failures = check_performance_thresholds(metrics)
    if not passed:
        failures.extend(perf_failures)

    logger.info("\n--- 2. Model Size ---")
    passed, msg = check_model_size(model_path)
    if not passed:
        failures.append(msg)

    logger.info("\n--- 3. Inference Speed ---")
    passed, msg = check_inference_speed(model_path)
    if not passed:
        failures.append(msg)

    logger.info("\n--- 4. Feature Schema ---")
    passed, msg = check_schema(metrics)
    if not passed:
        failures.append(msg)

    logger.info("\n--- 5. Prediction Sanity ---")
    passed, msg = check_prediction_sanity(model_path)
    if not passed:
        failures.append(msg)

    # ── Summary ───────────────────────────────────────────────
    logger.info("\n" + "="*60)
    if failures:
        logger.error("🚨 VALIDATION FAILED — MODEL REJECTED")
        logger.error("="*60)
        for f in failures:
            logger.error(f"  ❌ {f}")
        logger.error("\nThis model will NOT be deployed.")
        logger.error("Fix the issues above and retrain.\n")
        return False
    else:
        logger.info("✅ ALL VALIDATION CHECKS PASSED")
        logger.info("="*60)
        logger.info("Model approved for deployment.\n")
        return True


if __name__ == "__main__":
    passed = run_validation()
    sys.exit(0 if passed else 1)

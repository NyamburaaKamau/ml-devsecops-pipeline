# 🧠 ML DevSecOps Pipeline

> A complete, production-grade CI/CD pipeline for ML models - from training to deployment with security baked in at every stage. Not just a pipeline template. A full ML Platform Engineering story.

[![Pipeline](https://img.shields.io/github/actions/workflow/status/NyamburaaKamau/ml-devsecops-pipeline/pipeline.yml?label=Pipeline&logo=github-actions)](https://github.com/NyamburaaKamau/ml-devsecops-pipeline/actions)
[![Trivy](https://img.shields.io/badge/Trivy-0%20Critical%20CVEs-green)](https://github.com/aquasecurity/trivy)
[![Cosign](https://img.shields.io/badge/Cosign-Image%20Signed-blue)](https://github.com/sigstore/cosign)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange)](https://mlflow.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem This Solves

Most ML pipelines are either:

- **ML only** - Great model, no deployment story. Lives in a notebook forever.
- **DevOps only** - Great pipeline, no ML validation. Bad models ship to production.
- **Neither** - "SSH into EC2 and run python serve.py" 😬

This repo shows the **complete picture** - ML training and validation, platform-grade deployment, and security at every stage. Every layer is owned.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML DevSecOps Pipeline                        │
├──────────┬──────────────┬──────────────┬────────────────────────┤
│ 🧠 ML    │ 🔐 Security  │ 🐳 Platform  │ 🚀 Deploy              │
├──────────┼──────────────┼──────────────┼────────────────────────┤
│ Train    │ Gitleaks     │ Docker Build │ Verify Signature       │
│ Validate │ Semgrep SAST │ Trivy Scan   │ kubectl apply          │
│ Gate     │ tfsec        │ Cosign Sign  │ Smoke Test             │
│ MLflow   │ Checkov      │ SBOM Gen     │ Auto Rollback on Fail  │
└──────────┴──────────────┴──────────────┴────────────────────────┘
         ↓ FAIL FAST - any stage fails, pipeline stops ↓
```

---

## What's Inside

```
ml-devsecops-pipeline/
├── .github/workflows/
│   └── pipeline.yml          # Full 7-stage CI/CD pipeline
├── ml/
│   ├── model/
│   │   ├── train.py          # Model training + MLflow tracking
│   │   └── metrics.json      # Generated training metrics
│   └── validation/
│       └── validate.py       # Model validation gate (5 checks)
├── examples/sample-model/
│   └── serve.py              # FastAPI serving + Prometheus metrics
├── platform/
│   ├── kubernetes/
│   │   └── deployment.yaml   # Hardened K8s manifests + HPA
│   ├── terraform/            # IaC for cluster provisioning
│   └── monitoring/           # Prometheus rules + Grafana dashboards
├── security/
│   ├── policies/             # OPA Gatekeeper constraints
│   └── scanning/             # Trivy + Semgrep configs
├── Dockerfile                # Multi-stage, non-root, hardened
└── requirements.txt
```

---

## 🧠 ML Layer

### Model Training

Trains a **Gradient Boosting churn prediction model** with full MLflow experiment tracking. Replace with your actual model - the pipeline is model-agnostic.

```bash
python ml/model/train.py
```

Tracked in MLflow:

- Hyperparameters
- Training metrics (accuracy, ROC-AUC, F1, precision, recall)
- Cross-validation scores
- Model artifact + version

### Model Validation Gate

**This is what separates ML Platform Engineering from just DevOps.**

5 automated checks before any model can deploy:

| Check                  | What It Catches                                  |
| ---------------------- | ------------------------------------------------ |
| Performance thresholds | Models that don't meet accuracy/ROC-AUC minimums |
| Model size             | Accidentally bloated models                      |
| Inference speed        | Models too slow for production SLAs              |
| Feature schema         | Models trained on wrong features                 |
| Prediction sanity      | Degenerate models that predict only one class    |

```bash
python ml/validation/validate.py
# Exit 0 = approved ✅
# Exit 1 = rejected ❌ — pipeline stops here
```

Example output:

```
============================================================
🔍 RUNNING MODEL VALIDATION GATE
============================================================
--- 1. Performance Thresholds ---
✅ PASS: accuracy: 0.8340 >= 0.8000
✅ PASS: roc_auc: 0.8921 >= 0.8200
✅ PASS: f1_score: 0.8102 >= 0.7500

--- 2. Model Size ---
✅ PASS: Model size: 2.34MB (limit: 100MB)

--- 3. Inference Speed ---
✅ PASS: Inference speed: 12.3ms for 100 samples (limit: 100ms)

--- 4. Feature Schema ---
✅ PASS: all 9 required features present

--- 5. Prediction Sanity ---
✅ PASS: 2 classes, 34.50% positive rate

============================================================
✅ ALL VALIDATION CHECKS PASSED — Model approved for deployment
============================================================
```

---

## 🔐 Security Layer

Security gates run in **parallel** with model validation - no waiting.

### Stage 1: Secret Scanning (Gitleaks)

Scans full git history for leaked credentials, API keys, passwords. Pipeline fails immediately if any secrets detected. No exceptions.

### Stage 2: SAST (Semgrep)

Static analysis across Python code with:

- `p/python` - Python security rules
- `p/security-audit` - Security audit patterns
- `p/secrets` - Secret patterns in code
- `p/owasp-top-ten` - OWASP Top 10 checks

### Stage 3: IaC Scanning

- **tfsec** - Terraform misconfiguration detection
- **Checkov** - Kubernetes manifest security scanning
- Results uploaded as SARIF to GitHub Security tab

### Stage 4: Container Scanning (Trivy)

Scans the built Docker image for:

- OS package vulnerabilities
- Language dependency vulnerabilities
- CRITICAL/HIGH CVEs → pipeline fails

```
CRITICAL: 0 ✅
HIGH:      0 ✅
MEDIUM:    2 ⚠️  (accepted, non-exploitable)
```

### Stage 5: Image Signing (Cosign)

Every image is signed using **keyless signing via Sigstore** before deployment. The deploy stage **verifies the signature** before running `kubectl apply`. Unsigned images cannot be deployed.

```bash
# Verify an image signature yourself
cosign verify \
  --certificate-identity-regexp="https://github.com/NyamburaaKamau/*" \
  --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
  ghcr.io/nyambuaaKamau/ml-devsecops-pipeline/ml-serving@sha256:abc123
```

---

## ⚙️ Platform Layer

### FastAPI Model Server

- Async inference with Prometheus metrics built in
- Structured JSON logging
- Input validation via Pydantic
- Graceful shutdown on SIGTERM
- `/health` (liveness) + `/ready` (readiness) probes

### Kubernetes Deployment

- Non-root container (UID 1001)
- Read-only root filesystem
- All Linux capabilities dropped
- `automountServiceAccountToken: false`
- HPA: 2-10 pods based on CPU/memory
- Zero-downtime rolling updates (`maxUnavailable: 0`)

### Observability

Prometheus metrics tracked on every prediction:

- `ml_predictions_total` - by model version and result
- `ml_prediction_latency_seconds` - P50/P95/P99 histogram
- `ml_prediction_confidence` - confidence score distribution
- `ml_model_loaded` - model health gauge

---

## Running Locally

```bash
# Clone
git clone https://github.com/NyamburaaKamau/ml-devsecops-pipeline
cd ml-devsecops-pipeline

# Install dependencies
pip install -r requirements.txt

# Train model
python ml/model/train.py

# Validate model
python ml/validation/validate.py

# Serve model
uvicorn examples.sample-model.serve:app --reload --port 8000

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "monthly_charges": 75.5,
    "total_charges": 450.0,
    "num_products": 2,
    "support_calls": 4,
    "payment_delay_days": 5,
    "contract_length": 1,
    "has_online_backup": 0,
    "has_tech_support": 0
  }'
```

---

## Adapting for Your Model

1. Replace `ml/model/train.py` with your training code
2. Update `THRESHOLDS` in `ml/validation/validate.py` for your metrics
3. Update `REQUIRED_FEATURES` in `validate.py` for your feature schema
4. Replace `examples/sample-model/serve.py` with your serving logic
5. Push to `main` — pipeline runs automatically

---

## Key Design Decisions

**Why validate the model before building the image?**
Building a Docker image takes 2-3 minutes. Running validation takes 10 seconds. Fail fast - don't waste time building images for rejected models.

**Why keyless Cosign signing?**
No key management, no key rotation, no key storage. Sigstore uses the GitHub Actions OIDC token as the signing identity. Simpler and more secure.

**Why `maxUnavailable: 0` in rolling updates?**
ML serving needs zero downtime. During a rolling update, new pods must be fully ready before old pods are terminated. A model that's still loading shouldn't receive traffic.

**Why check prediction sanity?**
A model with 99% accuracy that predicts "no churn" for everyone has learned the class imbalance, not the pattern. The sanity check catches degenerate models that performance metrics alone can miss.

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Cosign Keyless Signing](https://docs.sigstore.dev/cosign/signing/overview/)
- [Trivy Container Scanning](https://aquasecurity.github.io/trivy/)
- [Semgrep Rules](https://semgrep.dev/explore)
- [Kubernetes Rolling Updates](https://kubernetes.io/docs/tutorials/kubernetes-basics/update/update-intro/)

---

\*Built with 🧠 by [Nyambura Kamau](https://github.com/NyamburaaKamau)

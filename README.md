# Customer Churn Prediction — Telco

End-to-end machine learning solution that predicts which telecom customers are about to churn,
so the retention team can reach them first.

**Business Problem → Data Preparation → EDA → Feature Engineering → Model → Evaluation →
Interpretation → Saved Pipeline → REST API**

**Business Problem → Data Preparation → EDA → Feature Engineering → Model → Evaluation →
Interpretation → Saved Pipeline → REST API**

**Repository:** https://github.com/saurabhr76/customer-churn-project-NAGP

---


---

## Results at a glance

Final model: **Decision Tree** (`max_depth=6`, `min_samples_leaf=80`, `class_weight="balanced"`,
`random_state=42`), evaluated on **2,113 hold-out customers** never seen during training.

| Metric | Score | What it means for the business |
|---|---|---|
| **Recall** | **0.7790** | We catch **~78% of customers who actually churn** — the metric that matters most |
| **Precision** | **0.5052** | ~half the call list is a real churner — **1.9× better than random** (base rate 26.5%) |
| **F1 Score** | **0.6129** | Best of every tree configuration tested |
| **ROC-AUC** | **0.8277** | Reliable risk *ranking* for prioritising the call list |
| **Accuracy** | **0.7388** | Deliberately traded down for recall — see the notebook, §5.3 |

Confusion matrix (test set):

|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 1,124 (TN) | 428 (FP — wasted offer, cheap) |
| **Actual: Churn** | 124 (FN — **lost customer, expensive**) | 437 (TP — saved in time) |

**Top churn drivers:** `Contract = Month-to-month` (61% of model importance) ›
`ChargePerService` *(engineered)* (16%) › `tenure` (8%) › `AvgMonthlySpend` *(engineered)* (4%) ›
`PaymentMethod = Electronic check` (3%).

---

## Project structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv                 # 7,043 customers × 21 columns
│   └── TelcoCustomerChurn_DataDictionary.csv
├── notebook/
│   └── churn_analysis.ipynb                   # complete analysis (49 code + 62 markdown cells, 16 figures)
├── model/
│   └── churn_model.pkl                        # saved bundle: pipeline + threshold + metrics + metadata
├── churn_features.py                          # shared cleaning + feature engineering (notebook AND API import this)
├── app.py                                     # FastAPI service (POST /predict)
├── requirements.txt
├── sample_request.json
├── sample_response.json
└── README.md
```

---

## Setup and execution

### 1. Install

```bash
cd customer_churn_project
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+ required (developed and verified on 3.12).

### 2. Run the analysis notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

Run all cells. It reproduces every number in this README and re-saves `model/churn_model.pkl`.
All randomness is seeded with `random_state=42`, so results are identical on any machine.

### 3. Start the API

```bash
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | **`/predict`** | Score one customer → prediction + probability |
| `POST` | `/predict/batch` | Score up to 1,000 customers in one call |
| `GET` | `/health` | Liveness + "is the model loaded" probe |
| `GET` | `/model-info` | Version, hyper-parameters, hold-out metrics, expected schema |
| `GET` | `/docs` | Interactive Swagger UI |

### Sample request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 94.40,
  "TotalCharges": 188.80
}
```

### Sample response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8903,
  "risk_level": "Very High",
  "threshold_used": 0.5,
  "model_version": "1.0.0"
}
```

A long-tenure, two-year-contract customer returns:

```json
{
  "prediction": "No",
  "churn_probability": 0.0906,
  "risk_level": "Low",
  "threshold_used": 0.5,
  "model_version": "1.0.0"
}
```

### Invalid input handling

| Case | Response |
|---|---|
| Missing required field | `422` with `{"detail":[{"loc":["body","tenure"],"msg":"Field required"}]}` |
| Invalid category (`"Contract": "Monthly"`) | `422`, listing the allowed values |
| Wrong type (`"MonthlyCharges": "ninety"`) | `422` |
| Out of range (`"tenure": -5`) | `422` |
| Unknown category at prediction time | Handled gracefully — `OneHotEncoder(handle_unknown="ignore")` |
| `TotalCharges` blank/null (brand-new customer) | Accepted — reconstructed as `tenure × MonthlyCharges` |
| Model file missing | `503` with a clear message |

---

## Key technical decisions

**1. One `Pipeline` from raw JSON to probability.**
Feature engineering → imputation → one-hot encoding → the tree are a single `sklearn.Pipeline`
object. `joblib.dump()` saves the whole path, so **the API re-implements no transformation** —
that is what eliminates training/serving skew.

**2. No data leakage, by construction.**
Every fitted transformer (imputer, encoder) lives *inside* the pipeline, so it is only ever
fitted on the training fold — including inside each cross-validation fold. All 8 engineered
features are **row-wise and stateless** (no column means, no target statistics), so they can be
computed for a single API payload exactly as they were in training.

**3. `TotalCharges` repair rule.**
11 rows hold a blank string; all 11 have `tenure = 0` (customers who have not been billed yet).
Rather than dropping them or imputing a meaningless median, the value is reconstructed as
`tenure × MonthlyCharges`, which yields the true value 0. Deterministic and applicable to unseen data.

**4. `class_weight="balanced"` for the 73/27 imbalance.**
Without it, the tree maximises accuracy by rarely predicting churn (recall ≈ 0.42). With it,
recall rises to ≈ 0.78 — the single most important modelling decision in the project.

**5. The threshold is stored in the artifact, not hard-coded.**
The API derives its Yes/No label from `bundle["threshold"]`. When the retention budget or the
estimated customer lifetime value changes, the business re-tunes one number — **no retraining**.

---

## Precision or Recall for churn?

**Recall**, decisively — subject to a precision floor the call-centre budget can absorb.

A **false negative** (a churner we never call) costs the **entire customer lifetime value** plus
the cost of acquiring a replacement — industry rule of thumb, 5–7× the cost of retention — and it
is **irreversible**. A **false positive** (a retention offer to a loyal customer) costs one
discount and a few minutes of agent time, is **reversible**, and often builds goodwill anyway.
The expensive error is roughly **10×** the cheap one, so the metric that counts the expensive
error is the one to optimise.

The nuance: recall cannot be pushed to 1.0, because flagging everyone is the same as having no
model. The real objective is *maximise recall subject to call-centre capacity and a precision
floor* — formalised in the notebook (§5.4) as an expected-cost minimisation over the decision
threshold, which lands well below 0.5 and cuts expected churn cost by **~80% versus doing nothing**.

---

## Business recommendations

| Priority | Action | Evidence |
|---|---|---|
| 1 | **Migrate month-to-month customers to annual contracts** | 61% of model importance; churn 42.7% → 11.3% (1yr) → 2.8% (2yr) |
| 2 | **Own the first 90 days** with a structured onboarding programme | >50% churn in months 0–3 |
| 3 | **Incentivise auto-pay enrolment** | Electronic check 45.3% vs automatic methods ~16% |
| 4 | **Bundle TechSupport + OnlineSecurity into fiber plans** | No support ~42% vs with support ~15% |
| 5 | **Review fiber price/value for new customers** | Fiber 41.9% vs DSL 19.0%; new + high-ARPU >60% |
"""
churn_features.py
=================
Shared, single-source-of-truth data cleaning + feature engineering logic for the
Telco Customer Churn project.

WHY THIS FILE EXISTS
--------------------
The exact same transformations must run in three places:
    1. the training notebook,
    2. the saved sklearn pipeline (joblib pickles a *reference* to these
       functions, not their code), and
    3. the FastAPI service that scores new customers.

Keeping them in one importable module guarantees training/serving consistency
(no "training-serving skew") and means the pickle can always be un-pickled,
because `churn_features` is importable from the project root.

DESIGN RULE — NO DATA LEAKAGE
-----------------------------
Every function below is **row-wise and stateless**: the value computed for a
customer depends only on that customer's own attributes, never on statistics
learned from other rows (no means, no target encoding, no global ranks).
Anything that *must* learn from the data (median imputation, one-hot vocabulary)
is done by fitted scikit-learn transformers *inside* the Pipeline, so it is
fitted on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column groups (raw schema)
# --------------------------------------------------------------------------- #
ID_COL = "customerID"
TARGET = "Churn"

#: The six optional add-on services offered on top of the internet subscription.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

#: Services that protect / support the customer (as opposed to entertainment).
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]

#: Entertainment / streaming services.
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]

#: Raw numeric columns.
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

#: Raw categorical columns (SeniorCitizen is 0/1 but is semantically categorical).
RAW_CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

#: Engineered numeric columns produced by `engineer_features`.
ENGINEERED_NUMERIC = [
    "AvgMonthlySpend",
    "ChargeGrowthRatio",
    "NumAddOnServices",
    "NumProtectionServices",
    "ChargePerService",
    "ExpectedLifetimeValue",
]

#: Engineered categorical columns produced by `engineer_features`.
ENGINEERED_CATEGORICAL = [
    "TenureGroup",
    "IsAutoPayment",
    "HasFamily",
    "IsNewCustomer",
    "HasStreaming",
]

#: Final feature lists consumed by the ColumnTransformer.
NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL

#: Every column the model expects to receive from the caller (API contract).
MODEL_INPUT_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL


# --------------------------------------------------------------------------- #
# 1. Raw cleaning
# --------------------------------------------------------------------------- #
def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Repair the known data-quality defects of the Telco dataset.

    Fixes applied
    -------------
    * ``TotalCharges`` arrives as text because 11 rows contain a blank string
      (" "). It is coerced to numeric; the blanks become NaN.
    * Those 11 rows all have ``tenure == 0`` — brand-new customers who have not
      been billed yet. Instead of dropping them (losing real customers) or using
      a global median (nonsense for a day-one customer), we reconstruct the value
      from the business identity ``TotalCharges ≈ tenure × MonthlyCharges``,
      which yields 0 for tenure-0 customers. This rule is deterministic and
      therefore safe to apply to unseen data too.
    * ``customerID`` is dropped: it is a unique key with zero predictive signal
      and would let a tree memorise individual customers.
    * Whitespace is stripped from string columns.

    The function is idempotent and safe to run on a single-row API payload.
    """
    out = df.copy()

    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])

    # Strip stray whitespace on all text columns.
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) in ("str", "string"):
            out[col] = out[col].astype("object").str.strip()

    # TotalCharges -> numeric, then rule-based repair.
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if "tenure" in out.columns and "MonthlyCharges" in out.columns:
            reconstructed = (
                pd.to_numeric(out["tenure"], errors="coerce")
                * pd.to_numeric(out["MonthlyCharges"], errors="coerce")
            )
            out["TotalCharges"] = out["TotalCharges"].fillna(reconstructed)
        out["TotalCharges"] = out["TotalCharges"].fillna(0.0)

    # SeniorCitizen is 0/1 — keep it as a string label so it is one-hot encoded
    # like every other binary attribute (and so the API can accept 0/1 or "Yes").
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = (
            out["SeniorCitizen"]
            .replace({0: "No", 1: "Yes", "0": "No", "1": "Yes"})
            .astype("object")
        )

    return out


# --------------------------------------------------------------------------- #
# 2. Feature engineering (row-wise, stateless -> leakage-proof)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features described in section 3 of the report.

    All features are derived from a single customer's own record, so this step
    can sit *inside* the sklearn Pipeline and be applied identically to the
    training set, the test set and a live API request.
    """
    out = clean_raw_dataframe(df)

    tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0)
    total = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0)

    # --- F1: average amount actually billed per month of tenure -------------
    # Smooths one-off promos/discounts and reveals the customer's *realised*
    # spend level rather than the current rate-card price.
    safe_tenure = tenure.replace(0, np.nan)
    out["AvgMonthlySpend"] = (total / safe_tenure).fillna(monthly).round(4)

    # --- F2: current bill vs. historical average ("bill shock" detector) ----
    # Ratio > 1 means the customer is currently paying more than they have on
    # average -> a price increase or an add-on upsell, a classic churn trigger.
    denom = out["AvgMonthlySpend"].replace(0, np.nan)
    out["ChargeGrowthRatio"] = (monthly / denom).fillna(1.0).clip(0, 5).round(4)

    # --- F3: how many optional services the customer holds ------------------
    # Each extra service raises switching cost ("product stickiness").
    addon_flags = pd.DataFrame(
        {c: (out[c].astype(str) == "Yes").astype(int) for c in ADDON_SERVICES},
        index=out.index,
    )
    out["NumAddOnServices"] = addon_flags.sum(axis=1)

    # --- F4: protection/support bundle count --------------------------------
    # Security/backup/protection/tech-support specifically correlate with a
    # supported, low-friction experience, unlike entertainment add-ons.
    out["NumProtectionServices"] = addon_flags[PROTECTION_SERVICES].sum(axis=1)

    # --- F5: price paid per service held ("value for money") ----------------
    # High monthly charge spread over few services = poor perceived value.
    base_services = (out["PhoneService"].astype(str) == "Yes").astype(int) + (
        out["InternetService"].astype(str) != "No"
    ).astype(int)
    out["ChargePerService"] = (monthly / (out["NumAddOnServices"] + base_services + 1)).round(4)

    # --- F6: expected contract-horizon value --------------------------------
    # Monthly bill scaled by the commitment length: what the customer is worth
    # if they see out their current contract. Useful for cost-of-churn framing.
    contract_months = out["Contract"].map(
        {"Month-to-month": 1, "One year": 12, "Two year": 24}
    ).fillna(1).astype(float)
    out["ExpectedLifetimeValue"] = (monthly * contract_months).round(2)

    # --- F7: tenure lifecycle stage -----------------------------------------
    # Churn hazard is strongly non-linear in tenure; explicit buckets let the
    # tree split on lifecycle stage in one clean step.
    out["TenureGroup"] = pd.cut(
        tenure,
        bins=[-0.1, 6, 12, 24, 48, 60, 1000],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "60m+"],
    ).astype(str)

    # --- F8: behavioural / demographic binary flags --------------------------
    out["IsAutoPayment"] = np.where(
        out["PaymentMethod"].astype(str).str.contains("automatic", case=False, na=False),
        "Yes",
        "No",
    )
    out["HasFamily"] = np.where(
        (out["Partner"].astype(str) == "Yes") | (out["Dependents"].astype(str) == "Yes"),
        "Yes",
        "No",
    )
    out["IsNewCustomer"] = np.where(tenure <= 6, "Yes", "No")
    out["HasStreaming"] = np.where(
        addon_flags[STREAMING_SERVICES].sum(axis=1) > 0, "Yes", "No"
    )

    return out


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns the ColumnTransformer expects, in a stable order."""
    eng = engineer_features(df)
    return eng[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

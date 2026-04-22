# -*- coding: utf-8 -*-
"""
============================================================
African Credit Scoring Challenge - Senior ML Pipeline
============================================================
Author : Senior ML Engineer
Date   : 2026-04

Dataset Context
---------------
- Train: Kenya loans only
- Test : Kenya + Ghana loans -> model must generalise cross-country
- Each row = one lender's share of a loan
  (same loan_id can appear with multiple lender_ids)
- Economic indicators from FRED (per country, per year)

Strategy
--------
1. Rigorous EDA with class-imbalance diagnostics
2. Rich feature engineering (temporal, financial ratios,
   multi-lender aggregations, country economic indicators)
3. No target leakage - feature engineering on combined data,
   labels never visible to test set
4. Cross-validated ensemble: LightGBM + XGBoost (+ CatBoost)
5. Threshold optimisation on OOF predictions (F1)
6. SampleSubmission-compatible output
"""

# ============================================================
# 0. Stdout - force ASCII-safe UTF-8 on Windows
# ============================================================
import sys, io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                  encoding="utf-8", errors="replace")

# ============================================================
# 1. Auto-install missing packages
# ============================================================
import subprocess, importlib

def _ensure(pip_name, import_name=None):
    mod = import_name or pip_name
    try:
        return importlib.import_module(mod)
    except ImportError:
        print(f"  pip install {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               pip_name, "-q"], stderr=subprocess.DEVNULL)
        return importlib.import_module(mod)

_ensure("lightgbm")
_ensure("xgboost")
_ensure("scikit-learn", "sklearn")

# ============================================================
# 2. Core imports
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend - safe on all platforms
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_recall_curve, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb

# CatBoost is optional
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("[OK] CatBoost available")
except ImportError:
    print("  CatBoost not installed - running LGB + XGB only.")

print(f"\nPython  : {sys.version.split()[0]}")
print(f"LightGBM: {lgb.__version__}")
print(f"XGBoost : {xgb.__version__}")
print(f"NumPy   : {np.__version__}")
print(f"Pandas  : {pd.__version__}")

# ============================================================
# 3. Paths & constants
# ============================================================
DATA_DIR        = Path(__file__).parent
TRAIN_PATH      = DATA_DIR / "Train.csv"
TEST_PATH       = DATA_DIR / "Test.csv"
ECON_PATH       = DATA_DIR / "economic_indicators.csv"
SUBMISSION_PATH = DATA_DIR / "SampleSubmission.csv"
OUTPUT_PATH     = DATA_DIR / "submission.csv"

RANDOM_STATE = 42
N_FOLDS      = 5

# ============================================================
# 4. Load data
# ============================================================
print("\n" + "=" * 60)
print("Loading data ...")
train               = pd.read_csv(TRAIN_PATH)
test                = pd.read_csv(TEST_PATH)
econ_raw            = pd.read_csv(ECON_PATH)
submission_template = pd.read_csv(SUBMISSION_PATH)

print(f"  Train shape : {train.shape}")
print(f"  Test  shape : {test.shape}")
print(f"  Submission  : {submission_template.shape}")
print(f"  Target dist :\n{train['target'].value_counts(normalize=True).round(4)}")

# ============================================================
# 5. EDA
# ============================================================
print("\n" + "=" * 60)
print("EDA ...")
imb = (train["target"] == 0).sum() / max((train["target"] == 1).sum(), 1)
print(f"  Class imbalance ratio (0:1)  = {imb:.1f}x")
print(f"  Train missing values         = {train.isna().sum().sum()}")
print(f"  Test  missing values         = {test.isna().sum().sum()}")
print(f"  Unique loan_types  (train)   = {train['loan_type'].nunique()}")
print(f"  Countries  (train)           = {sorted(train['country_id'].unique())}")
print(f"  Countries  (test)            = {sorted(test['country_id'].unique())}")
print(f"  Unique lender_ids (train)    = {train['lender_id'].nunique()}")
print(f"  Unique customer_ids (train)  = {train['customer_id'].nunique()}")

# ============================================================
# 6. Economic indicators
# ============================================================
print("\n" + "=" * 60)
print("Preparing economic indicators ...")

econ_melted = econ_raw.melt(
    id_vars=["Country", "Indicator"],
    var_name="Year", value_name="Value",
)
econ_melted["Year"] = (
    econ_melted["Year"].str.replace("YR", "", regex=False).astype(int)
)

econ_pivot = (
    econ_melted
    .pivot_table(index=["Country", "Year"],
                 columns="Indicator", values="Value",
                 aggfunc="first")
    .reset_index()
)
econ_pivot.columns.name = None

RENAME_ECON = {
    "Inflation, consumer prices (annual %)":               "econ_inflation",
    "Official exchange rate (LCU per US$, period average)":"econ_exchange_rate",
    "Real interest rate (%)":                              "econ_real_interest",
    "Average precipitation in depth (mm per year)":        "econ_precipitation",
    "Deposit interest rate (%)":                           "econ_deposit_rate",
    "Lending interest rate (%)":                           "econ_lending_rate",
    "Interest rate spread (lending rate minus deposit rate, %)": "econ_rate_spread",
    "Fossil fuel energy consumption (% of total)":         "econ_fossil_fuel",
    "Unemployment rate":                                   "econ_unemployment",
}
econ_pivot.rename(columns=RENAME_ECON, inplace=True)
econ_pivot.rename(columns={"Country": "country_econ", "Year": "year_econ"},
                  inplace=True)

ECON_COLS = list(RENAME_ECON.values())
print(f"  Economic feature columns: {ECON_COLS}")


def merge_econ(df: pd.DataFrame) -> pd.DataFrame:
    """Merge macro indicators by (country_id, disbursement year)."""
    tmp = df.copy()
    tmp["_year"] = pd.to_datetime(tmp["disbursement_date"]).dt.year
    out = tmp.merge(
        econ_pivot[["country_econ", "year_econ"] + ECON_COLS],
        left_on=["country_id", "_year"],
        right_on=["country_econ", "year_econ"],
        how="left",
    )
    out.drop(columns=["_year", "country_econ", "year_econ"],
             errors="ignore", inplace=True)
    return out

# ============================================================
# 7. Multi-lender aggregations (computed on ALL data - no leakage)
# ============================================================
print("Building multi-lender loan aggregations ...")
all_raw = pd.concat([train, test], axis=0, ignore_index=True)

loan_agg = (
    all_raw.groupby("tbl_loan_id", sort=False)
    .agg(
        loan_n_lenders       =("lender_id",               "nunique"),
        loan_total_funded    =("Amount_Funded_By_Lender",  "sum"),
        loan_max_share       =("Lender_portion_Funded",    "max"),
        loan_min_share       =("Lender_portion_Funded",    "min"),
        loan_std_share       =("Lender_portion_Funded",    "std"),
    )
    .reset_index()
)
loan_agg["loan_std_share"].fillna(0.0, inplace=True)

# Customer history - derived from TRAIN only (no leakage to test labels)
cust_agg = (
    train.groupby("customer_id", sort=False)
    .agg(
        cust_n_loans      =("tbl_loan_id",  "nunique"),
        cust_avg_amount   =("Total_Amount", "mean"),
        cust_max_amount   =("Total_Amount", "max"),
        cust_default_rate =("target",       "mean"),   # label mean - train only
    )
    .reset_index()
)

# Lender reliability stats from TRAIN
lender_agg = (
    train.groupby("lender_id", sort=False)
    .agg(
        lender_n_loans      =("tbl_loan_id",  "nunique"),
        lender_default_rate =("target",       "mean"),
    )
    .reset_index()
)

# ============================================================
# 8. Feature engineering function
# ============================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates
    disb = pd.to_datetime(df["disbursement_date"])
    due  = pd.to_datetime(df["due_date"])

    df["disb_year"]          = disb.dt.year          # kept for econ merge key
    df["disb_month"]         = disb.dt.month
    df["disb_day"]           = disb.dt.day
    df["disb_weekday"]       = disb.dt.weekday
    df["disb_quarter"]       = disb.dt.quarter
    df["disb_week"]          = disb.dt.isocalendar().week.astype(int)
    df["disb_dayofyear"]     = disb.dt.dayofyear
    df["disb_is_weekend"]    = (df["disb_weekday"] >= 5).astype(np.int8)
    df["disb_is_monthstart"] = disb.dt.is_month_start.astype(np.int8)
    df["disb_is_monthend"]   = disb.dt.is_month_end.astype(np.int8)

    df["due_month"]          = due.dt.month
    df["due_weekday"]        = due.dt.weekday
    df["due_quarter"]        = due.dt.quarter
    df["due_is_weekend"]     = (df["due_weekday"] >= 5).astype(np.int8)

    df["loan_term_days"]     = (due - disb).dt.days
    df["term_vs_duration"]   = df["loan_term_days"] / df["duration"].clip(lower=1)

    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[0, 7, 14, 30, 60, 90, 180, 365, np.inf],
        labels=[0, 1, 2, 3, 4, 5, 6, 7],
    ).astype(int)

    # Financial ratios
    eps = 1e-6
    A   = df["Total_Amount"].clip(lower=eps)
    R   = df["Total_Amount_to_Repay"]
    LF  = df["Amount_Funded_By_Lender"]
    LP  = df["Lender_portion_Funded"]
    LR  = df["Lender_portion_to_be_repaid"]
    DUR = df["duration"].clip(lower=1)

    df["repayment_ratio"]    = R / A
    df["interest_amount"]    = (R - df["Total_Amount"]).clip(lower=0)
    df["interest_pct"]       = df["interest_amount"] / A
    df["lender_share"]       = LF / A
    df["lender_repay_ratio"] = LR / (LF.clip(lower=eps))
    df["borrower_portion"]   = (1.0 - LP).clip(lower=0.0, upper=1.0)
    df["borrower_amount"]    = df["Total_Amount"] * df["borrower_portion"]
    df["borrower_repay"]     = (R - LR).clip(lower=0)
    df["daily_repayment"]    = R / DUR
    df["daily_lender_repay"] = LR / DUR
    df["log_loan"]           = np.log1p(df["Total_Amount"])
    df["log_repay"]          = np.log1p(R)
    df["log_lender_funded"]  = np.log1p(LF)

    df["loan_size_bucket"] = pd.cut(
        df["Total_Amount"],
        bins=[0, 1000, 5000, 20000, 50000, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # Categorical
    df["is_repeat_loan"] = (
        df["New_versus_Repeat"].str.strip() == "Repeat Loan"
    ).astype(np.int8)

    df["loan_type_code"] = LabelEncoder().fit_transform(
        df["loan_type"].astype(str)
    )

    # Interaction features
    df["amount_x_duration"]    = df["Total_Amount"] * df["duration"]
    df["interest_x_duration"]  = df["interest_amount"] * df["duration"]
    df["lender_share_x_loan"]  = df["lender_share"] * df["log_loan"]
    df["repay_ratio_x_term"]   = df["repayment_ratio"] * df["loan_term_days"]

    return df


# ============================================================
# 9. Build full feature matrix
# ============================================================
print("\n" + "=" * 60)
print("Building feature matrix ...")

combined = pd.concat([train, test], axis=0, ignore_index=True)
combined = merge_econ(combined)
combined = feature_engineering(combined)
combined = combined.merge(loan_agg,  on="tbl_loan_id",  how="left")
combined = combined.merge(cust_agg,  on="customer_id",  how="left")
combined = combined.merge(lender_agg, on="lender_id",   how="left")

# Fill customer/lender stats for test rows not seen in train
FILL_COLS = ["cust_n_loans", "cust_avg_amount", "cust_max_amount",
             "cust_default_rate", "lender_n_loans", "lender_default_rate"]
train_mask = combined["ID"].isin(train["ID"])
for col in FILL_COLS:
    med = combined.loc[train_mask, col].median()
    combined[col] = combined[col].fillna(med)

# Drop raw text/date columns
combined.drop(
    columns=["disbursement_date", "due_date",
             "New_versus_Repeat", "loan_type"],
    inplace=True,
)

# One-hot encode country (low cardinality)
combined = pd.get_dummies(combined, columns=["country_id"],
                          drop_first=False, dtype=np.int8)

# Re-split
train_fe = combined[combined["ID"].isin(train["ID"])].copy()
test_fe  = combined[~combined["ID"].isin(train["ID"])].copy()

# Feature columns
EXCLUDE = {
    "ID", "target",
    "customer_id", "tbl_loan_id", "lender_id",   # high-cardinality IDs
    "disb_year",                                  # used only as merge key
}
FEATURES = [c for c in train_fe.columns if c not in EXCLUDE]

# Encode remaining object columns
for col in train_fe[FEATURES].select_dtypes(include="object").columns:
    le = LabelEncoder()
    le.fit(pd.concat([train_fe[col], test_fe[col]]).astype(str))
    train_fe[col] = le.transform(train_fe[col].astype(str))
    test_fe[col]  = le.transform(test_fe[col].astype(str))

# Fill any remaining NaN with train-median
for col in FEATURES:
    if train_fe[col].isna().any() or test_fe[col].isna().any():
        med = float(train_fe[col].median())
        train_fe[col] = train_fe[col].fillna(med)
        test_fe[col]  = test_fe[col].fillna(med)

# Cast to float32 - required by XGBoost >= 2.0 DMatrix
X      = train_fe[FEATURES].astype(np.float32).values
y      = train_fe["target"].astype(np.int32).values
X_test = test_fe[FEATURES].astype(np.float32).values

print(f"  Feature count   : {len(FEATURES)}")
print(f"  Train matrix    : {X.shape}")
print(f"  Test  matrix    : {X_test.shape}")
print(f"  Positive rate   : {y.mean():.4f}")

# ============================================================
# 10. Model parameters
# ============================================================
scale_pos_weight = float((y == 0).sum()) / float(max((y == 1).sum(), 1))
print(f"\n  scale_pos_weight = {scale_pos_weight:.2f}")

lgb_params = dict(
    objective        = "binary",
    metric           = "auc",
    learning_rate    = 0.05,
    num_leaves       = 127,
    max_depth        = -1,
    min_child_samples= 50,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 0.5,
    is_unbalance     = True,
    random_state     = RANDOM_STATE,
    n_estimators     = 3000,
    verbosity        = -1,
    n_jobs           = -1,
)

# XGBoost >= 2.0: use_label_encoder removed
# XGBoost 3.x: early_stopping_rounds is back in the constructor
xgb_base_params = dict(
    objective          = "binary:logistic",
    eval_metric        = "auc",
    learning_rate      = 0.05,
    max_depth          = 6,
    n_estimators       = 3000,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    scale_pos_weight   = scale_pos_weight,
    seed               = RANDOM_STATE,
    verbosity          = 0,
    n_jobs             = -1,
    tree_method        = "hist",
    device             = "cpu",
    early_stopping_rounds = 150,   # XGBoost 3.x: in constructor
)

if CATBOOST_AVAILABLE:
    cat_params = dict(
        iterations           = 3000,
        learning_rate        = 0.05,
        depth                = 6,
        l2_leaf_reg          = 3,
        auto_class_weights   = "Balanced",
        random_seed          = RANDOM_STATE,
        early_stopping_rounds= 150,
        verbose              = 0,
    )

# ============================================================
# 11. Cross-validated training
# ============================================================
print("\n" + "=" * 60)
print(f"Running {N_FOLDS}-fold stratified cross-validation ...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                      random_state=RANDOM_STATE)

oof_lgb  = np.zeros(len(X), dtype=np.float64)
oof_xgb  = np.zeros(len(X), dtype=np.float64)
oof_cat  = np.zeros(len(X), dtype=np.float64)
test_lgb = np.zeros(len(X_test), dtype=np.float64)
test_xgb = np.zeros(len(X_test), dtype=np.float64)
test_cat = np.zeros(len(X_test), dtype=np.float64)

lgb_models, xgb_models, cat_models = [], [], []

for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n  -- Fold {fold_i}/{N_FOLDS} --")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    oof_lgb[va_idx]  = lgb_model.predict_proba(X_va)[:, 1]
    test_lgb        += lgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    lgb_models.append(lgb_model)
    print(f"    LGB  AUC={roc_auc_score(y_va, oof_lgb[va_idx]):.4f}"
          f"  best_iter={lgb_model.best_iteration_}")

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_base_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    oof_xgb[va_idx]  = xgb_model.predict_proba(X_va)[:, 1]
    test_xgb        += xgb_model.predict_proba(X_test)[:, 1] / N_FOLDS
    xgb_models.append(xgb_model)
    print(f"    XGB  AUC={roc_auc_score(y_va, oof_xgb[va_idx]):.4f}"
          f"  best_iter={xgb_model.best_iteration}")

    # CatBoost (optional)
    if CATBOOST_AVAILABLE:
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_tr, y_tr,
                      eval_set=(X_va, y_va), use_best_model=True)
        oof_cat[va_idx]  = cat_model.predict_proba(X_va)[:, 1]
        test_cat        += cat_model.predict_proba(X_test)[:, 1] / N_FOLDS
        cat_models.append(cat_model)
        print(f"    CAT  AUC={roc_auc_score(y_va, oof_cat[va_idx]):.4f}")

# ============================================================
# 12. Ensemble (AUC-weighted average)
# ============================================================
auc_lgb = roc_auc_score(y, oof_lgb)
auc_xgb = roc_auc_score(y, oof_xgb)
auc_cat = roc_auc_score(y, oof_cat) if CATBOOST_AVAILABLE else 0.0

print("\n" + "=" * 60)
print("OOF AUC scores:")
print(f"  LightGBM : {auc_lgb:.5f}")
print(f"  XGBoost  : {auc_xgb:.5f}")
if CATBOOST_AVAILABLE:
    print(f"  CatBoost : {auc_cat:.5f}")

total_w  = auc_lgb + auc_xgb + (auc_cat if CATBOOST_AVAILABLE else 0.0)
w_lgb    = auc_lgb / total_w
w_xgb    = auc_xgb / total_w
w_cat    = (auc_cat / total_w) if CATBOOST_AVAILABLE else 0.0

oof_ens  = w_lgb * oof_lgb  + w_xgb * oof_xgb  + w_cat * oof_cat
test_ens = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

print(f"\n  Weights  LGB={w_lgb:.3f}  XGB={w_xgb:.3f}  CAT={w_cat:.3f}")
print(f"  Ensemble OOF AUC : {roc_auc_score(y, oof_ens):.5f}")

# ============================================================
# 13. Threshold optimisation (maximise F1)
# ============================================================
print("\n" + "=" * 60)
print("Optimising classification threshold for F1 ...")

precisions, recalls, thr_vals = precision_recall_curve(y, oof_ens)
f1_arr   = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = int(np.argmax(f1_arr))
best_thr = float(thr_vals[best_idx])
best_f1  = float(f1_arr[best_idx])

print(f"  Optimal threshold : {best_thr:.4f}")
print(f"  OOF F1            : {best_f1:.5f}")
print(f"  OOF AUC           : {roc_auc_score(y, oof_ens):.5f}")

y_pred_oof = (oof_ens >= best_thr).astype(int)
print("\nOOF Classification Report:")
print(classification_report(y, y_pred_oof, digits=4))

# ============================================================
# 14. Plots
# ============================================================
# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y, y_pred_oof,
                                        ax=ax, colorbar=False)
ax.set_title(f"OOF Confusion Matrix  (thr={best_thr:.4f})")
plt.tight_layout()
plt.savefig(DATA_DIR / "oof_confusion_matrix.png", dpi=150)
plt.close()
print("\n  Saved: oof_confusion_matrix.png")

# Feature importance (LightGBM avg across folds)
fi = pd.DataFrame({
    "feature":    FEATURES,
    "importance": np.mean(
        [m.feature_importances_ for m in lgb_models], axis=0
    ),
}).sort_values("importance", ascending=False).reset_index(drop=True)

top_n = min(35, len(fi))
fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(data=fi.head(top_n), y="feature", x="importance",
            palette="viridis", ax=ax)
ax.set_title("LightGBM - Avg Feature Importance (across folds)")
plt.tight_layout()
plt.savefig(DATA_DIR / "feature_importance.png", dpi=150)
plt.close()
print("  Saved: feature_importance.png")
print("\nTop 20 features:")
print(fi.head(20).to_string(index=False))

# Precision-Recall curve
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls, precisions, lw=1.5, color="steelblue", label="PR curve")
ax.scatter(recalls[best_idx], precisions[best_idx],
           color="red", zorder=5,
           label=f"Opt thr={best_thr:.3f}  F1={best_f1:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("OOF Precision-Recall Curve")
ax.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / "pr_curve.png", dpi=150)
plt.close()
print("  Saved: pr_curve.png")

# ============================================================
# 15. Generate submission
# ============================================================
print("\n" + "=" * 60)
print("Generating submission ...")

y_test_pred  = (test_ens >= best_thr).astype(int)
id_to_pred   = dict(zip(test_fe["ID"].values, y_test_pred))

submission   = submission_template.copy()
submission["target"] = submission["ID"].map(id_to_pred)

n_missing = int(submission["target"].isna().sum())
if n_missing > 0:
    print(f"  WARNING: {n_missing} IDs have no prediction -- filling with 0")
    submission["target"] = submission["target"].fillna(0)

submission["target"] = submission["target"].astype(int)
submission.to_csv(OUTPUT_PATH, index=False)

print(f"  Submission rows       : {len(submission)}")
print(f"  Predicted default (1) : {(submission['target'] == 1).sum()}")
print(f"  Predicted non-default : {(submission['target'] == 0).sum()}")
print(f"  Default rate in sub   : {submission['target'].mean():.4f}")
print(f"\n  Saved -> {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"  [OK] submission.csv")
print(f"  [OK] oof_confusion_matrix.png")
print(f"  [OK] feature_importance.png")
print(f"  [OK] pr_curve.png")
print("=" * 60)

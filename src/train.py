"""
Customer Churn Prediction using Machine Learning
================================================
Training script (Part 3).

What this script does, in order:
  1. Loads the IBM Telco Customer Churn CSV
  2. Inspects the data (shape, types, missing values, duplicates, class balance)
  3. Cleans the data (TotalCharges -> number, duplicates, ID column, target to 0/1)
  4. Saves EDA graphs
  5. Splits the data 80/20 (stratified, random_state=42)
  6. Builds ONE leakage-safe Pipeline (ColumnTransformer + Logistic Regression)
  7. Trains on the training set only and evaluates on the test set
  8. Saves figures, metrics, and the complete trained pipeline

Run from the project folder (Windows PowerShell):
    python src\\train.py
Optional:
    python src\\train.py --balanced        (uses class_weight="balanced")

Every number printed or saved is computed from the data. Nothing is hard-coded.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files; never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# 1. Settings (paths are built from this file's location, so the script works
#    no matter which folder you run it from)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
# All 19 input features, in the same order as the CSV file
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
EXPECTED_COLUMNS = [ID_COL] + FEATURE_COLUMNS + [TARGET]
assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES), \
    "numeric + categorical lists must cover exactly the 19 feature columns"

LOG_LINES = []  # everything we print is also saved to outputs/metrics/run_log.txt


def log(message=""):
    """Print a message and keep it for the run log file."""
    print(message)
    LOG_LINES.append(str(message))


# ----------------------------------------------------------------------------
# 2. Load and inspect
# ----------------------------------------------------------------------------
def load_data():
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at:\n  {DATA_PATH}")
        print("Download it from the project folder with this PowerShell command:")
        print('  Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IBM/'
              'telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" '
              '-OutFile "data\\raw\\Telco-Customer-Churn.csv"')
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: these expected columns are missing from the CSV: {missing_cols}")
        sys.exit(1)
    return df


def inspect_data(df):
    log("=" * 70)
    log("STEP 1: DATA INSPECTION (raw data)")
    log("=" * 70)
    log(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    log("\nColumn types:")
    log(df.dtypes.astype(str).to_string())
    log("\nMissing values per column (real NaN only):")
    nulls = df.isna().sum()
    log(nulls[nulls > 0].to_string() if nulls.sum() > 0 else "  none")

    blanks = {}
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            count = int((df[col].str.strip() == "").sum())
            if count > 0:
                blanks[col] = count
    log("\nBlank text entries (look like data, but are really missing):")
    log("\n".join(f"  {c}: {n}" for c, n in blanks.items()) if blanks else "  none")

    log(f"\nDuplicate rows (all columns): {df.duplicated().sum()}")
    log(f"Duplicate {ID_COL} values:     {df[ID_COL].duplicated().sum()}")
    log("\nTarget counts:")
    log(df[TARGET].value_counts().to_string())
    log("\nTarget percentages:")
    log((df[TARGET].value_counts(normalize=True) * 100).round(2).to_string())


# ----------------------------------------------------------------------------
# 3. Clean
# ----------------------------------------------------------------------------
def clean_data(df):
    log("\n" + "=" * 70)
    log("STEP 2: DATA CLEANING")
    log("=" * 70)
    df = df.copy()

    # 3a. Remove stray spaces in text columns; empty text becomes a real NaN
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].str.strip()
    df = df.replace("", np.nan)

    # 3b. TotalCharges is text in the CSV (blank for brand-new customers).
    #     errors="coerce" turns anything that is not a number into NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    bad_total = int(df["TotalCharges"].isna().sum())
    log(f"TotalCharges converted to numbers. Blank or non-numeric entries now missing: {bad_total}")
    if bad_total > 0:
        tenure_of_missing = sorted(int(t) for t in df.loc[df["TotalCharges"].isna(), "tenure"].unique())
        log(f"  tenure of those rows: {tenure_of_missing} "
            "(they are filled later by the pipeline imputer, using training data only)")

    # 3c. Duplicates
    n = len(df)
    df = df.drop_duplicates()
    log(f"Duplicate rows removed: {n - len(df)}")
    n = len(df)
    df = df.drop_duplicates(subset=ID_COL, keep="first")
    log(f"Rows removed for repeated {ID_COL}: {n - len(df)}")

    # 3d. Rows with no target cannot be used for training
    n = len(df)
    df = df.dropna(subset=[TARGET])
    log(f"Rows removed for missing {TARGET}: {n - len(df)}")

    # 3e. Target: "No"/"Yes" -> 0/1  (1 = customer churned)
    allowed = {"No", "Yes"}
    found = set(df[TARGET].unique())
    if not found.issubset(allowed):
        print(f"ERROR: unexpected {TARGET} values: {found - allowed}")
        sys.exit(1)
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    # 3f. The ID column is a label, not a feature. Remove it.
    df = df.drop(columns=[ID_COL])

    remaining = df.isna().sum()
    log("\nMissing values left after cleaning (handled inside the pipeline):")
    log(remaining[remaining > 0].to_string() if remaining.sum() > 0 else "  none")
    log(f"\nClean data shape: {df.shape}")
    return df


# ----------------------------------------------------------------------------
# 4. EDA graphs
#    EDA only describes the data. No value learned here is used by the model.
# ----------------------------------------------------------------------------
def save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {path.relative_to(PROJECT_ROOT)}")


def churn_rate_plot(df, column, filename, title, horizontal=False):
    """Bar chart of the churn percentage inside each category of one column."""
    rate = (df.groupby(column)[TARGET].mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if horizontal:
        sns.barplot(x=rate.values, y=rate.index, hue=rate.index, palette="viridis",
                    legend=False, ax=ax)
        ax.set_xlabel("Churn rate (%)")
        ax.set_ylabel(column)
        for i, v in enumerate(rate.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    else:
        sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="viridis",
                    legend=False, ax=ax)
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel(column)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.set_title(title)
    save_fig(fig, filename)


def make_eda_plots(df):
    log("\n" + "=" * 70)
    log("STEP 3: EDA GRAPHS")
    log("=" * 70)
    sns.set_theme(style="whitegrid")
    plot_df = df.copy()
    plot_df["Churn label"] = plot_df[TARGET].map({0: "No", 1: "Yes"})

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(data=plot_df, x="Churn label", hue="Churn label", order=["No", "Yes"],
                  palette=["#4c9f70", "#d9534f"], legend=False, ax=ax)
    total = len(plot_df)
    for patch in ax.patches:
        count = int(patch.get_height())
        ax.annotate(f"{count} ({count / total * 100:.1f}%)",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    save_fig(fig, "01_churn_distribution.png")

    # 2-4. Churn rate by important categories
    churn_rate_plot(df, "Contract", "02_churn_by_contract.png", "Churn Rate by Contract Type")
    churn_rate_plot(df, "InternetService", "03_churn_by_internet_service.png",
                    "Churn Rate by Internet Service")
    churn_rate_plot(df, "PaymentMethod", "04_churn_by_payment_method.png",
                    "Churn Rate by Payment Method", horizontal=True)

    # 5. Tenure distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=plot_df, x="tenure", hue="Churn label", bins=36, multiple="layer",
                 palette={"No": "#4c9f70", "Yes": "#d9534f"}, alpha=0.55, ax=ax)
    ax.set_title("Tenure (months) by Churn")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Number of customers")
    save_fig(fig, "05_tenure_distribution.png")

    # 6. Monthly charges box plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=plot_df, x="Churn label", y="MonthlyCharges", hue="Churn label",
                order=["No", "Yes"], palette=["#4c9f70", "#d9534f"], legend=False, ax=ax)
    ax.set_title("Monthly Charges by Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly charges")
    save_fig(fig, "06_monthly_charges_boxplot.png")

    # 7. Correlation heatmap (numeric columns + target)
    corr_cols = NUMERIC_FEATURES + [TARGET]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of Numeric Features with Churn")
    save_fig(fig, "07_correlation_heatmap.png")


# ----------------------------------------------------------------------------
# 5. Pipeline (this is what makes preprocessing leakage-safe)
# ----------------------------------------------------------------------------
def build_pipeline(class_weight=None):
    """
    Imputers, the scaler, and the encoder live INSIDE the pipeline. When we call
    pipeline.fit(X_train, y_train) they learn their values (median, mean, std,
    categories) from the training rows only. The test rows are only transformed.
    """
    numeric_steps = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_steps, NUMERIC_FEATURES),
            ("cat", categorical_steps, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


# ----------------------------------------------------------------------------
# 6. Evaluation
# ----------------------------------------------------------------------------
def evaluate(pipeline, X_test, y_test):
    log("\n" + "=" * 70)
    log("STEP 6: EVALUATION ON THE TEST SET")
    log("=" * 70)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of churn (class 1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred, zero_division=0),
        "recall_churn": recall_score(y_test, y_pred, zero_division=0),
        "f1_churn": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "majority_class_baseline_accuracy": max(y_test.mean(), 1 - y_test.mean()),
        "test_rows": int(len(y_test)),
    }
    for name, value in metrics.items():
        if name == "test_rows":
            log(f"{name:34s}: {value}")
        else:
            log(f"{name:34s}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log("\nConfusion matrix (rows = actual, columns = predicted):")
    log(pd.DataFrame(cm, index=["Actual No", "Actual Yes"],
                     columns=["Pred No", "Pred Yes"]).to_string())
    log(f"\nTrue Negatives={tn}  False Positives={fp}  False Negatives={fn}  True Positives={tp}")

    report = classification_report(y_test, y_pred, target_names=["No churn", "Churn"],
                                   zero_division=0)
    log("\nClassification report:")
    log(report)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.grid(False)
    ax.set_title("Confusion Matrix (test set)")
    save_fig(fig, "08_confusion_matrix.png")

    # ROC curve figure
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"Logistic Regression (AUC = {metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve (test set)")
    ax.legend(loc="lower right")
    save_fig(fig, "09_roc_curve.png")

    # Text/JSON/CSV files
    (METRICS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=["Actual No", "Actual Yes"],
                 columns=["Pred No", "Pred Yes"]).to_csv(METRICS_DIR / "confusion_matrix.csv")
    return metrics


def save_coefficients(pipeline):
    """Save and plot the model's strongest features (readable for the report)."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    names = [n.replace("num__", "").replace("cat__", "") for n in names]
    coefs = pipeline.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({"feature": names, "coefficient": coefs})
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df = coef_df.sort_values("coefficient", ascending=False)
    coef_df.to_csv(METRICS_DIR / "feature_coefficients.csv", index=False)

    top = pd.concat([coef_df.head(10), coef_df.tail(10)])
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#d9534f" if c > 0 else "#4c9f70" for c in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (positive = more likely to churn)")
    ax.set_title("Top 10 Features Raising and Lowering Churn Risk")
    save_fig(fig, "10_top_feature_coefficients.png")

    log("\nTop 5 features that INCREASE churn risk:")
    log(coef_df.head(5)[["feature", "coefficient"]].round(3).to_string(index=False))
    log("\nTop 5 features that DECREASE churn risk:")
    log(coef_df.tail(5).iloc[::-1][["feature", "coefficient"]].round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the customer churn model.")
    parser.add_argument("--balanced", action="store_true",
                        help='use class_weight="balanced" in Logistic Regression')
    args = parser.parse_args()
    class_weight = "balanced" if args.balanced else None

    start = time.time()
    for folder in (FIG_DIR, METRICS_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    log(f"Python {platform.python_version()} | pandas {pd.__version__} | "
        f"numpy {np.__version__} | scikit-learn {sklearn.__version__}")
    log(f"Project folder: {PROJECT_ROOT}")
    log(f"class_weight: {class_weight}")

    raw = load_data()
    inspect_data(raw)
    df = clean_data(raw)
    make_eda_plots(df)

    # ---- Split FIRST, before any learning from the data ----
    log("\n" + "=" * 70)
    log("STEP 4: TRAIN / TEST SPLIT (80/20, stratified, random_state=42)")
    log("=" * 70)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log(f"Training rows: {len(X_train)}   Test rows: {len(X_test)}")
    log(f"Churn rate in full data: {y.mean() * 100:.2f}%  | train: "
        f"{y_train.mean() * 100:.2f}%  | test: {y_test.mean() * 100:.2f}%")

    # ---- Safety checks against leakage ----
    assert ID_COL not in X.columns, "customerID must not be a feature"
    assert TARGET not in X.columns, "the target must not be a feature"
    assert set(X_train.index).isdisjoint(set(X_test.index)), "train and test rows overlap"

    # ---- Train ----
    log("\n" + "=" * 70)
    log("STEP 5: TRAIN THE PIPELINE (fit on the training set only)")
    log("=" * 70)
    pipeline = build_pipeline(class_weight)
    pipeline.fit(X_train, y_train)
    n_out = len(pipeline.named_steps["preprocess"].get_feature_names_out())
    log(f"Pipeline trained. Input columns: {X_train.shape[1]} -> model features after encoding: {n_out}")
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    log(f"Training accuracy (for overfitting check): {train_acc:.4f}")

    metrics = evaluate(pipeline, X_test, y_test)
    metrics["train_accuracy"] = train_acc
    save_coefficients(pipeline)

    # ---- Save ----
    log("\n" + "=" * 70)
    log("STEP 7: SAVING FILES")
    log("=" * 70)
    joblib.dump(pipeline, MODEL_PATH)
    log(f"  saved {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    allowed_values = {col: sorted(df[col].dropna().unique().tolist())
                      for col in CATEGORICAL_FEATURES}
    model_info = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "allowed_categorical_values": allowed_values,
        "target_mapping": {"0": "No churn", "1": "Churn"},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "class_weight": class_weight,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    log(f"  saved {MODEL_INFO_PATH.relative_to(PROJECT_ROOT)}")

    metrics_out = {k: (float(v) if not isinstance(v, int) else v) for k, v in metrics.items()}
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log(f"  saved {(METRICS_DIR / 'metrics.json').relative_to(PROJECT_ROOT)}")

    log(f"\nDone in {time.time() - start:.1f} seconds.")
    (METRICS_DIR / "run_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()

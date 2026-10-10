from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"
]

def load_telco(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python download_data.py` first."
        )
    df = pd.read_csv(path)

    missing_cols = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_cols or extra_cols:
        raise ValueError(
            f"Schema mismatch. Missing={missing_cols}; Extra={extra_cols}"
        )

    # IBM sample includes blank TotalCharges values represented as whitespace.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    if df["Churn"].isna().any():
        raise ValueError("Unexpected target labels in Churn.")

    return df

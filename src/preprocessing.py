import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def load_raw_csv(path: str) -> pd.DataFrame:
    """
    Load raw FDR CSV safely.
    """
    return pd.read_csv(path, low_memory=False)


def drop_fully_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop sensors that are 100% missing.
    """
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio == 1.0].index
    return df.drop(columns=cols_to_drop)


def convert_mixed_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-type numeric columns (e.g. EGT sensors) to float.
    Non-convertible values become NaN.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep numeric columns only.
    """
    return df.select_dtypes(include="number")


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then backward-fill missing values.
    Standard practice for flight data.
    """
    return df.ffill().bfill()


def set_time_index(df_num: pd.DataFrame, time_series: pd.Series) -> pd.DataFrame:
    """
    Set Session Time as index.
    """
    df_num = df_num.copy()
    df_num["Session Time"] = time_series.values
    df_num = df_num.set_index("Session Time")
    return df_num


def scale_features(df_num: pd.DataFrame):
    """
    Robust scaling (safe for outliers).
    Returns scaled dataframe + scaler.
    """
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df_num.values)
    df_scaled = pd.DataFrame(
        scaled,
        index=df_num.index,
        columns=df_num.columns
    )
    return df_scaled, scaler


def preprocess_fdr(path: str):
    """
    Full preprocessing pipeline.
    """
    df = load_raw_csv(path)

    # Save time before dropping anything
    time_col = df["Session Time"]

    df = drop_fully_missing_columns(df)
    df = convert_mixed_numeric(df)

    df_num = select_numeric(df)
    df_num = fill_missing(df_num)
    df_num = set_time_index(df_num, time_col)

    df_scaled, scaler = scale_features(df_num)

    return {
        "raw": df,
        "numeric": df_num,
        "scaled": df_scaled,
        "scaler": scaler
    }

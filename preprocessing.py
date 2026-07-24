"""
Shared preprocessing module for Employee Attrition Prediction.

This module contains the single source of truth for all feature engineering
and preprocessing logic, used by both training (model.py) and serving (app.py).

Key design principle: NO DATA LEAKAGE
- Feature engineering (build_features) is stateless and can be applied to any data
- Encoder fitting happens ONLY on training data (after train/test split)
- Inference uses the pre-fitted encoder from the saved artifact
"""

from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# Categorical columns that need one-hot encoding (after feature engineering)
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Education",
    "EducationField",
    "MaritalStatus",
    "StockOptionLevel",
    "OverTime",
    "Gender",
    "TrainingTimesLastYear",
]

# Thresholds for boolean feature engineering (single source of truth)
THRESHOLDS = {
    "Total_Satisfaction": 2.8,
    "Age": 35,
    "DailyRate": 800,
    "Department_RnD": "Research & Development",
    "DistanceFromHome": 10,
    "JobRole_LabTech": "Laboratory Technician",
    "HourlyRate": 65,
    "MonthlyIncome": 4000,
    "NumCompaniesWorked": 3,
    "TotalWorkingYears": 8,
    "YearsAtCompany": 3,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 1,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to raw dataframe.

    This is the SINGLE SOURCE OF TRUTH for feature engineering.
    Used by both training and serving. Stateless - no fitting.

    Args:
        df: Raw dataframe with original columns

    Returns:
        DataFrame with engineered features (before one-hot encoding)
    """
    df = df.copy()

    # Drop unnecessary columns early
    cols_to_drop = [c for c in ["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Target encoding (if present)
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

    # OverTime encoding
    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)

    # Total Satisfaction feature
    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    if all(c in df.columns for c in satisfaction_cols):
        df["Total_Satisfaction"] = df[satisfaction_cols].mean(axis=1)
        df.drop(columns=satisfaction_cols, inplace=True)
        df["Total_Satisfaction_bool"] = df["Total_Satisfaction"].apply(
            lambda x: 1 if x >= THRESHOLDS["Total_Satisfaction"] else 0
        )
        df.drop(columns=["Total_Satisfaction"], inplace=True)

    # Boolean feature engineering
    boolean_features = [
        ("Age", "Age_bool", lambda x: 1 if x < THRESHOLDS["Age"] else 0),
        ("DailyRate", "DailyRate_bool", lambda x: 1 if x < THRESHOLDS["DailyRate"] else 0),
        ("Department", "Department_bool", lambda x: 1 if x == THRESHOLDS["Department_RnD"] else 0),
        ("DistanceFromHome", "DistanceFromHome_bool", lambda x: 1 if x > THRESHOLDS["DistanceFromHome"] else 0),
        ("JobRole", "JobRole_bool", lambda x: 1 if x == THRESHOLDS["JobRole_LabTech"] else 0),
        ("HourlyRate", "HourlyRate_bool", lambda x: 1 if x < THRESHOLDS["HourlyRate"] else 0),
        ("MonthlyIncome", "MonthlyIncome_bool", lambda x: 1 if x < THRESHOLDS["MonthlyIncome"] else 0),
        ("NumCompaniesWorked", "NumCompaniesWorked_bool", lambda x: 1 if x > THRESHOLDS["NumCompaniesWorked"] else 0),
        ("TotalWorkingYears", "TotalWorkingYears_bool", lambda x: 1 if x < THRESHOLDS["TotalWorkingYears"] else 0),
        ("YearsAtCompany", "YearsAtCompany_bool", lambda x: 1 if x < THRESHOLDS["YearsAtCompany"] else 0),
        ("YearsInCurrentRole", "YearsInCurrentRole_bool", lambda x: 1 if x < THRESHOLDS["YearsInCurrentRole"] else 0),
        ("YearsSinceLastPromotion", "YearsSinceLastPromotion_bool", lambda x: 1 if x < THRESHOLDS["YearsSinceLastPromotion"] else 0),
        ("YearsWithCurrManager", "YearsWithCurrManager_bool", lambda x: 1 if x < THRESHOLDS["YearsWithCurrManager"] else 0),
    ]

    for src_col, new_col, func in boolean_features:
        if src_col in df.columns:
            df[new_col] = df[src_col].apply(func)
            df.drop(columns=[src_col], inplace=True)

    # Gender encoding (Female=1, Male=0)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 0)

    # Drop columns not used
    drop_always = ["MonthlyRate", "PercentSalaryHike"]
    df.drop(columns=[c for c in drop_always if c in df.columns], inplace=True)

    return df


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert specified columns to categorical dtype."""
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "Attrition"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Split dataframe into features (categorical, numerical) and target."""
    df = prepare_categorical(df)

    X_categorical = df.select_dtypes(include=["category"])
    X_numerical = df.select_dtypes(include=["int64", "int32", "float64", "float32"])

    if target_col in X_numerical.columns:
        X_numerical = X_numerical.drop(columns=[target_col])

    y = df[target_col] if target_col in df.columns else None

    return X_categorical, X_numerical, y


def fit_encoder(X_categorical: pd.DataFrame) -> Tuple[OneHotEncoder, List[str]]:
    """
    Fit OneHotEncoder on categorical features and return encoder + feature names.

    This should ONLY be called on TRAINING data to avoid data leakage.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_categorical)

    feature_names = encoder.get_feature_names_out(X_categorical.columns.tolist()).tolist()

    return encoder, feature_names


def transform_features(
    X_categorical: pd.DataFrame,
    X_numerical: pd.DataFrame,
    encoder: OneHotEncoder,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Transform features using fitted encoder and combine with numerical.

    Returns DataFrame with columns in the exact order expected by the model.
    """
    # Transform categorical
    X_cat_encoded = encoder.transform(X_categorical)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X_categorical.index)

    # Ensure numerical columns are strings (to match training)
    X_num = X_numerical.copy()
    X_num.columns = X_num.columns.astype(str)

    # Concatenate
    X_all = pd.concat([X_cat_df, X_num], axis=1)

    return X_all


def preprocess_training_pipeline(
    df: pd.DataFrame, categorical_cols: list = None
) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder, List[str], List[str], List[str], List[str]]:
    """
    Full preprocessing pipeline for TRAINING data.

    This function:
    1. Applies feature engineering (stateless)
    2. Splits features/target
    3. Fits encoder ONLY on training data (no leakage)
    4. Returns everything needed for training and saving artifact

    Args:
        df: Raw training dataframe
        categorical_cols: List of categorical column names (default: CATEGORICAL_COLS)

    Returns:
        X_combined: Combined feature matrix (encoded cat + numerical)
        y: Target
        encoder: Fitted OneHotEncoder
        feature_names: All feature names in order
        numerical_columns: Numerical column names in order
        categorical_columns: Original categorical column names
        cat_feature_names: Encoded categorical feature names
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    # 1. Build features (stateless)
    df = build_features(df)

    # 2. Split features and target
    X_categorical, X_numerical, y = split_features_target(df)

    # 3. Ensure categorical columns exist and are in correct order
    for col in categorical_cols:
        if col not in X_categorical.columns:
            X_categorical[col] = pd.Series([0] * len(df), index=df.index, dtype="category")

    X_categorical = X_categorical[categorical_cols]

    # 4. Fit encoder on TRAINING data only (NO LEAKAGE)
    encoder, cat_feature_names = fit_encoder(X_categorical)

    # 5. Transform categorical
    X_cat_encoded = encoder.transform(X_categorical)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)

    # 6. Prepare numerical
    X_num = X_numerical.copy()
    X_num.columns = X_num.columns.astype(str)
    numerical_columns = X_num.columns.tolist()

    # 7. Combine
    X_combined = pd.concat([X_cat_df, X_num], axis=1)
    feature_names = X_combined.columns.tolist()

    return X_combined, y, encoder, feature_names, numerical_columns, categorical_cols, cat_feature_names


def save_model_artifact(
    model,
    encoder: OneHotEncoder,
    feature_names: List[str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    cat_feature_names: List[str],
    filepath: Union[str, Path] = "model.pkl",
) -> None:
    """
    Save model, encoder, and column metadata as a single artifact.

    This ensures exact schema match between training and serving.
    """
    artifact = {
        "model": model,
        "encoder": encoder,
        "feature_names": feature_names,  # all feature names in order (cat + num)
        "numerical_columns": numerical_columns,  # numerical column names in order
        "categorical_columns": categorical_columns,  # original categorical column names
        "cat_feature_names": cat_feature_names,  # encoded categorical feature names
        "thresholds": THRESHOLDS,
    }

    with open(filepath, "wb") as f:
        pickle.dump(artifact, f)


def load_model_artifact(filepath: Union[str, Path] = "model.pkl") -> Dict:
    """
    Load the complete model artifact.

    Returns dict with keys: model, encoder, feature_names, numerical_columns,
    categorical_columns, cat_feature_names, thresholds
    """
    with open(filepath, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def preprocess_for_inference(
    input_data: Dict,
    artifact: Dict,
) -> pd.DataFrame:
    """
    Preprocess a single input sample for inference using fitted artifact.

    Args:
        input_data: Raw input dictionary from form/API
        artifact: Loaded model artifact containing encoder, feature names, etc.

    Returns:
        DataFrame with exactly the same columns and order as training
    """
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])

    # Apply feature engineering (same as training - stateless)
    df = build_features(df)

    # Split into categorical and numerical
    X_categorical, X_numerical, _ = split_features_target(df)

    # Extract artifacts
    encoder = artifact["encoder"]
    numerical_columns = artifact["numerical_columns"]
    categorical_columns = artifact["categorical_columns"]
    cat_feature_names = artifact["cat_feature_names"]

    # Ensure categorical columns exist and are in correct order
    for col in categorical_columns:
        if col not in X_categorical.columns:
            X_categorical[col] = pd.Series([0], index=df.index, dtype="category")

    X_categorical = X_categorical[categorical_columns]

    # Ensure numerical columns exist and are in correct order
    for col in numerical_columns:
        if col not in X_numerical.columns:
            X_numerical[col] = 0

    X_numerical = X_numerical[numerical_columns]

    # Transform categorical using fitted encoder
    X_cat_encoded = encoder.transform(X_categorical)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)
    X_numerical.columns = X_numerical.columns.astype(str)

    # Combine
    X_combined = pd.concat([X_cat_df, X_numerical], axis=1)

    # Ensure columns match exactly what model was trained on
    expected_columns = cat_feature_names + numerical_columns
    X_combined = X_combined.reindex(columns=expected_columns, fill_value=0)

    return X_combined


# For backwards compatibility / standalone testing
if __name__ == "__main__":
    import sys
    data_path = Path("Data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        df = build_features(df)
        X_cat, X_num, y = split_features_target(df)
        print(f"Categorical shape: {X_cat.shape}")
        print(f"Numerical shape: {X_num.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Categorical columns: {X_cat.columns.tolist()}")
        print(f"Numerical columns: {X_num.columns.tolist()}")
    else:
        print(f"Data not found at {data_path}", file=sys.stderr)
        sys.exit(1)
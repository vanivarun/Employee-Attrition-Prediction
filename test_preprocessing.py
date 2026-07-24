"""
Unit tests for preprocessing module.
Run with: python -m pytest test_preprocessing.py -v
   or: python test_preprocessing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as prep


# Sample raw data matching training data format
SAMPLE_RAW_ROW = {
    "Age": 30,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 5,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Laboratory Technician",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "NumCompaniesWorked": 2,
    "OverTime": "Yes",
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 5,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 3,
    "YearsInCurrentRole": 2,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
    # Extra columns that get dropped
    "EmployeeNumber": 123,
    "Over18": "Y",
    "StandardHours": 80,
    "EmployeeCount": 1,
    "MonthlyRate": 5000,
    "PercentSalaryHike": 15,
    "Attrition": "No",
}


def test_build_features_basic():
    """Test that build_features produces expected columns and transformations."""
    df = pd.DataFrame([SAMPLE_RAW_ROW])
    result = prep.build_features(df)

    # Check Attrition is encoded (0/1), not dropped
    assert "Attrition" in result.columns
    assert result["Attrition"].iloc[0] == 0

    # Check unnecessary columns dropped
    assert "EmployeeNumber" not in result.columns
    assert "Over18" not in result.columns
    assert "StandardHours" not in result.columns
    assert "EmployeeCount" not in result.columns
    assert "MonthlyRate" not in result.columns
    assert "PercentSalaryHike" not in result.columns

    # Check boolean features created
    assert "Age_bool" in result.columns
    assert "DailyRate_bool" in result.columns
    assert "Department_bool" in result.columns
    assert "DistanceFromHome_bool" in result.columns
    assert "JobRole_bool" in result.columns
    assert "HourlyRate_bool" in result.columns
    assert "MonthlyIncome_bool" in result.columns
    assert "NumCompaniesWorked_bool" in result.columns
    assert "TotalWorkingYears_bool" in result.columns
    assert "YearsAtCompany_bool" in result.columns
    assert "YearsInCurrentRole_bool" in result.columns
    assert "YearsSinceLastPromotion_bool" in result.columns
    assert "YearsWithCurrManager_bool" in result.columns
    assert "Total_Satisfaction_bool" in result.columns

    # Check original columns dropped after boolean creation
    assert "Age" not in result.columns
    assert "DailyRate" not in result.columns
    assert "Department" not in result.columns
    assert "DistanceFromHome" not in result.columns
    assert "JobRole" not in result.columns
    assert "HourlyRate" not in result.columns
    assert "MonthlyIncome" not in result.columns
    assert "NumCompaniesWorked" not in result.columns
    assert "TotalWorkingYears" not in result.columns
    assert "YearsAtCompany" not in result.columns
    assert "YearsInCurrentRole" not in result.columns
    assert "YearsSinceLastPromotion" not in result.columns
    assert "YearsWithCurrManager" not in result.columns

    # Check satisfaction columns merged into Total_Satisfaction_bool
    for col in ["EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]:
        assert col not in result.columns

    # Check Gender encoded
    assert "Gender" in result.columns
    assert result["Gender"].iloc[0] in [0, 1]

    # Check OverTime encoded
    assert "OverTime" in result.columns
    assert result["OverTime"].iloc[0] in [0, 1]

    print(f"  Output columns: {list(result.columns)}")
    print(f"  Shape: {result.shape}")


def test_build_features_thresholds():
    """Test that boolean features use correct thresholds from THRESHOLDS."""
    df = pd.DataFrame([SAMPLE_RAW_ROW])

    # Test Age_bool: 1 if Age < 35
    result = prep.build_features(df)
    assert result["Age_bool"].iloc[0] == 1  # 30 < 35

    df2 = pd.DataFrame([{**SAMPLE_RAW_ROW, "Age": 40}])
    result2 = prep.build_features(df2)
    assert result2["Age_bool"].iloc[0] == 0  # 40 >= 35

    # Test DailyRate_bool: 1 if DailyRate < 800
    df3 = pd.DataFrame([{**SAMPLE_RAW_ROW, "DailyRate": 700}])
    result3 = prep.build_features(df3)
    assert result3["DailyRate_bool"].iloc[0] == 1

    df4 = pd.DataFrame([{**SAMPLE_RAW_ROW, "DailyRate": 900}])
    result4 = prep.build_features(df4)
    assert result4["DailyRate_bool"].iloc[0] == 0

    # Test Department_bool: 1 if R&D
    assert result["Department_bool"].iloc[0] == 1

    df5 = pd.DataFrame([{**SAMPLE_RAW_ROW, "Department": "Sales"}])
    result5 = prep.build_features(df5)
    assert result5["Department_bool"].iloc[0] == 0

    # Test MonthlyIncome_bool: 1 if MonthlyIncome < 4000
    df6 = pd.DataFrame([{**SAMPLE_RAW_ROW, "MonthlyIncome": 3000}])
    result6 = prep.build_features(df6)
    assert result6["MonthlyIncome_bool"].iloc[0] == 1

    df7 = pd.DataFrame([{**SAMPLE_RAW_ROW, "MonthlyIncome": 5000}])
    result7 = prep.build_features(df7)
    assert result7["MonthlyIncome_bool"].iloc[0] == 0


def test_build_features_categorical_preserved():
    """Test that categorical columns for one-hot encoding are present (some become numeric)."""
    df = pd.DataFrame([SAMPLE_RAW_ROW])
    result = prep.build_features(df)

    for col in prep.CATEGORICAL_COLS:
        assert col in result.columns, f"Missing categorical column: {col}"
        # After build_features: BusinessTravel, EducationField, MaritalStatus remain object.
        # Education, StockOptionLevel, OverTime, Gender, TrainingTimesLastYear become numeric.
        if col in ["BusinessTravel", "EducationField", "MaritalStatus"]:
            assert result[col].dtype == "object"
        else:
            assert result[col].dtype in ["int64", "int32"]


def test_prepare_categorical():
    """Test prepare_categorical converts CATEGORICAL_COLS to category dtype."""
    df = pd.DataFrame([SAMPLE_RAW_ROW])
    df = prep.build_features(df)
    result = prep.prepare_categorical(df)

    for col in prep.CATEGORICAL_COLS:
        if col in result.columns:
            assert result[col].dtype == "category"


def test_fit_encoder_and_transform():
    """Test encoder fitting and transformation produces expected columns."""
    df = pd.DataFrame([SAMPLE_RAW_ROW] * 10)  # Need multiple rows for encoder
    df = prep.build_features(df)
    X_cat, X_num, _ = prep.split_features_target(df)

    encoder, feature_names = prep.fit_encoder(X_cat)

    # Check encoder is fitted
    assert hasattr(encoder, "categories_")

    # Check feature names match expected pattern
    assert len(feature_names) > 0
    assert all(isinstance(n, str) for n in feature_names)

    # Transform
    X_transformed = prep.transform_features(X_cat, X_num, encoder, feature_names)
    assert list(X_transformed.columns) == feature_names + list(X_num.columns.astype(str))
    assert X_transformed.shape[0] == 10


def test_preprocess_for_inference_with_artifact():
    """Test full inference pipeline using a mock artifact."""
    # Create a minimal mock artifact similar to what model.py saves
    df = pd.DataFrame([SAMPLE_RAW_ROW] * 20)
    df = prep.build_features(df)
    X_cat, X_num, y = prep.split_features_target(df)

    for col in prep.CATEGORICAL_COLS:
        if col not in X_cat.columns:
            X_cat[col] = pd.Series([0] * len(df), index=df.index, dtype="category")
    X_cat = X_cat[prep.CATEGORICAL_COLS]

    encoder, cat_feature_names = prep.fit_encoder(X_cat)
    X_cat_encoded = encoder.transform(X_cat)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)
    X_num.columns = X_num.columns.astype(str)
    X_combined = pd.concat([X_cat_df, X_num], axis=1)
    feature_names = X_combined.columns.tolist()
    numerical_columns = X_num.columns.tolist()
    categorical_columns = prep.CATEGORICAL_COLS

    artifact = {
        "encoder": encoder,
        "feature_names": feature_names,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "cat_feature_names": cat_feature_names,
        "thresholds": prep.THRESHOLDS,
    }

    # Now test inference with form field names (matching HTML form)
    form_input = {
        "Age": 30,
        "BusinessTravel": "Travel_Rarely",
        "Daily Rate": 800,
        "Department": "Research & Development",
        "Distance From Home": 5,
        "Education": 3,
        "Education Field": "Life Sciences",
        "Environment Satisfaction": 3,
        "Gender": "Male",
        "Hourly Rate": 65,
        "Job Involvement": 3,
        "Job Level": 2,
        "Job Role": "Laboratory Technician",
        "Job Satisfaction": 3,
        "Marital Status": "Single",
        "Monthly Income": 5000,
        "Number of Companies Worked in": 2,
        "Over Time": "Yes",
        "Performance Rating": 3,
        "Relationship Satisfaction": 3,
        "Stock Option Level": 1,
        "Total Working Years": 5,
        "Training Times Last Year": 2,
        "Work Life Balance": 3,
        "Years At Company": 3,
        "Years In Current Role": 2,
        "Years Since Last Promotion": 1,
        "Years With Curr Manager": 2,
    }

    # This mimics what app.py does - parse form data to internal names
    field_mapping = {
        "BusinessTravel": "BusinessTravel",
        "Department": "Department",
        "Education Field": "EducationField",
        "Gender": "Gender",
        "Job Role": "JobRole",
        "Marital Status": "MaritalStatus",
        "Over Time": "OverTime",
    }
    num_fields = {
        "Age": "Age",
        "Daily Rate": "DailyRate",
        "Distance From Home": "DistanceFromHome",
        "Education": "Education",
        "Hourly Rate": "HourlyRate",
        "Job Level": "JobLevel",
        "Monthly Income": "MonthlyIncome",
        "Number of Companies Worked in": "NumCompaniesWorked",
        "Performance Rating": "PerformanceRating",
        "Stock Option Level": "StockOptionLevel",
        "Total Working Years": "TotalWorkingYears",
        "Training Times Last Year": "TrainingTimesLastYear",
        "Work Life Balance": "WorkLifeBalance",
        "Years At Company": "YearsAtCompany",
        "Years In Current Role": "YearsInCurrentRole",
        "Years Since Last Promotion": "YearsSinceLastPromotion",
        "Years With Curr Manager": "YearsWithCurrManager",
        "Environment Satisfaction": "EnvironmentSatisfaction",
        "Job Involvement": "JobInvolvement",
        "Job Satisfaction": "JobSatisfaction",
        "Relationship Satisfaction": "RelationshipSatisfaction",
    }

    internal = {}
    for form_name, internal_name in field_mapping.items():
        internal[internal_name] = form_input[form_name]
    for form_name, internal_name in num_fields.items():
        internal[internal_name] = int(form_input[form_name])

    X = prep.preprocess_for_inference(internal, artifact)

    # Verify output
    assert X.shape == (1, len(feature_names))
    assert list(X.columns) == feature_names
    assert not X.isnull().any().any()


def test_preprocess_for_inference_handles_unseen_categories():
    """Test that handle_unknown='ignore' works for unseen categorical values."""
    # Build artifact with known categories
    df = pd.DataFrame([SAMPLE_RAW_ROW] * 20)
    df = prep.build_features(df)
    X_cat, X_num, y = prep.split_features_target(df)

    for col in prep.CATEGORICAL_COLS:
        if col not in X_cat.columns:
            X_cat[col] = pd.Series([0] * len(df), index=df.index, dtype="category")
    X_cat = X_cat[prep.CATEGORICAL_COLS]

    encoder, cat_feature_names = prep.fit_encoder(X_cat)
    X_cat_encoded = encoder.transform(X_cat)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)
    X_num.columns = X_num.columns.astype(str)
    X_combined = pd.concat([X_cat_df, X_num], axis=1)
    feature_names = X_combined.columns.tolist()
    numerical_columns = X_num.columns.tolist()
    categorical_columns = prep.CATEGORICAL_COLS

    artifact = {
        "encoder": encoder,
        "feature_names": feature_names,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "cat_feature_names": cat_feature_names,
        "thresholds": prep.THRESHOLDS,
    }

    # Input with unseen BusinessTravel value
    internal = {
        "BusinessTravel": "Teleportation",  # Not in training data
        "Department": "Research & Development",
        "EducationField": "Life Sciences",
        "Gender": "Male",
        "JobRole": "Laboratory Technician",
        "MaritalStatus": "Single",
        "OverTime": "Yes",
        "Age": 30,
        "DailyRate": 800,
        "DistanceFromHome": 5,
        "Education": 3,
        "HourlyRate": 65,
        "JobLevel": 2,
        "MonthlyIncome": 5000,
        "NumCompaniesWorked": 2,
        "PerformanceRating": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
        "EnvironmentSatisfaction": 3,
        "JobInvolvement": 3,
        "JobSatisfaction": 3,
        "RelationshipSatisfaction": 3,
    }

    # Should not raise - handle_unknown='ignore' produces all zeros for that feature
    X = prep.preprocess_for_inference(internal, artifact)
    assert X.shape == (1, len(feature_names))


def test_preprocess_for_inference_missing_numerical_filled():
    """Test that missing numerical columns get filled with 0."""
    df = pd.DataFrame([SAMPLE_RAW_ROW] * 20)
    df = prep.build_features(df)
    X_cat, X_num, y = prep.split_features_target(df)

    for col in prep.CATEGORICAL_COLS:
        if col not in X_cat.columns:
            X_cat[col] = pd.Series([0] * len(df), index=df.index, dtype="category")
    X_cat = X_cat[prep.CATEGORICAL_COLS]

    encoder, cat_feature_names = prep.fit_encoder(X_cat)
    X_cat_encoded = encoder.transform(X_cat)
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=df.index)
    X_num.columns = X_num.columns.astype(str)
    X_combined = pd.concat([X_cat_df, X_num], axis=1)
    feature_names = X_combined.columns.tolist()
    numerical_columns = X_num.columns.tolist()
    categorical_columns = prep.CATEGORICAL_COLS

    artifact = {
        "encoder": encoder,
        "feature_names": feature_names,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "cat_feature_names": cat_feature_names,
        "thresholds": prep.THRESHOLDS,
    }

    # Provide only a subset of numerical fields
    internal = {
        "BusinessTravel": "Travel_Rarely",
        "Department": "Research & Development",
        "EducationField": "Life Sciences",
        "Gender": "Male",
        "JobRole": "Laboratory Technician",
        "MaritalStatus": "Single",
        "OverTime": "Yes",
        "Age": 30,
        "DailyRate": 800,
        # Missing many numerical fields
    }

    X = prep.preprocess_for_inference(internal, artifact)
    assert X.shape == (1, len(feature_names))
    # Raw numerical fields that don't go through build_features should be 0
    # (boolean features get computed from build_features so they have values)
    raw_numerical_fields = ["JobLevel", "PerformanceRating"]
    for col in raw_numerical_fields:
        assert X[col].iloc[0] == 0, f"Expected {col}=0, got {X[col].iloc[0]}"


def test_thresholds_constant():
    """Test that THRESHOLDS dict has all expected keys."""
    expected_keys = [
        "Total_Satisfaction", "Age", "DailyRate", "Department_RnD",
        "DistanceFromHome", "JobRole_LabTech", "HourlyRate", "MonthlyIncome",
        "NumCompaniesWorked", "TotalWorkingYears", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
    ]
    for key in expected_keys:
        assert key in prep.THRESHOLDS


def test_categorical_cols_constant():
    """Test that CATEGORICAL_COLS has expected columns."""
    expected = [
        "BusinessTravel", "Education", "EducationField", "MaritalStatus",
        "StockOptionLevel", "OverTime", "Gender", "TrainingTimesLastYear"
    ]
    assert prep.CATEGORICAL_COLS == expected


if __name__ == "__main__":
    # Run tests manually if pytest not available
    import traceback

    tests = [
        test_build_features_basic,
        test_build_features_thresholds,
        test_build_features_categorical_preserved,
        test_prepare_categorical,
        test_fit_encoder_and_transform,
        test_preprocess_for_inference_with_artifact,
        test_preprocess_for_inference_handles_unseen_categories,
        test_preprocess_for_inference_missing_numerical_filled,
        test_thresholds_constant,
        test_categorical_cols_constant,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"Running {test.__name__}...")
            test()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
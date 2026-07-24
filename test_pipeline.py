#!/usr/bin/env python
"""
Smoke test for Employee Attrition Prediction pipeline.

This script:
1. Trains the model end-to-end via the shared preprocessing module
2. Loads the saved model artifact back from disk
3. Builds a hand-crafted sample row using FORM FIELD NAMES (matching HTML form)
   and runs it through preprocessing + prediction
4. Asserts prediction is returned (0 or 1) with no exceptions
5. Verifies feature columns match exactly what the model was trained on

Run with: python test_pipeline.py
Exit code: 0 on success, non-zero on failure
"""

import sys
import traceback
import pickle
import subprocess
import os

import numpy as np
import pandas as pd

import preprocessing as prep


def train_model():
    """Train the model and save artifact."""
    print("=" * 60)
    print("STEP 1: Training model...")
    print("=" * 60)

    # Import and run model training
    result = subprocess.run([sys.executable, "model.py"], capture_output=True, text=True)

    if result.returncode != 0:
        print("Model training failed:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Model training failed")

    print(result.stdout)

    # Verify artifact was saved
    assert os.path.exists("model.pkl"), "model.pkl not created!"
    print("+ model.pkl created")

    # Load and return the artifact for further testing
    with open("model.pkl", "rb") as f:
        artifact = pickle.load(f)

    return artifact


def load_artifact():
    """Load the saved model artifact."""
    print("\n" + "=" * 60)
    print("STEP 2: Loading saved artifact...")
    print("=" * 60)

    with open("model.pkl", "rb") as f:
        artifact = pickle.load(f)

    # Verify artifact structure
    required_keys = ["model", "encoder", "feature_names", "numerical_columns", "categorical_columns", "thresholds"]
    for key in required_keys:
        assert key in artifact, f"Missing key in artifact: {key}"

    print(f"+ Artifact loaded: {type(artifact['model']).__name__}")
    print(f"+ Categorical features (encoded): {len(artifact['feature_names'])}")
    print(f"+ Numerical features: {len(artifact['numerical_columns'])}")
    print(f"+ Original categorical columns: {artifact['categorical_columns']}")

    return artifact


def create_sample_input():
    """
    Create a realistic sample input row matching the FORM FIELD NAMES (not internal names).
    This mirrors what the HTML form would submit.
    """
    return {
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


def parse_form_data(form_data: dict) -> dict:
    """Parse form data with proper type conversion and field name mapping.
    This mirrors app.py's parse_form_data function.
    """
    field_mapping = {
        "BusinessTravel": "BusinessTravel",
        "Department": "Department",
        "Education Field": "EducationField",
        "Gender": "Gender",
        "Job Role": "JobRole",
        "Marital Status": "MaritalStatus",
        "Over Time": "OverTime",
    }

    input_data = {}

    # Categorical fields
    for form_name, internal_name in field_mapping.items():
        input_data[internal_name] = form_data.get(form_name, "")

    # Numerical fields
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

    for form_name, internal_name in num_fields.items():
        try:
            input_data[internal_name] = int(form_data.get(form_name, 0))
        except (ValueError, TypeError):
            input_data[internal_name] = 0

    return input_data


def test_inference(artifact):
    """Run inference on sample input using the shared preprocessing."""
    print("\n" + "=" * 60)
    print("STEP 3: Running inference on sample input (form field names)...")
    print("=" * 60)

    sample = create_sample_input()
    print(f"Sample input keys: {list(sample.keys())}")

    # Parse form data to internal names (like app.py does)
    parsed = parse_form_data(sample)
    print(f"Parsed input keys: {list(parsed.keys())}")

    # Preprocess using shared function
    X = prep.preprocess_for_inference(parsed, artifact)

    print(f"+ Preprocessed feature matrix shape: {X.shape}")
    print(f"+ Feature columns: {list(X.columns)}")

    # Verify column order matches training
    expected_columns = artifact["feature_names"]
    actual_columns = list(X.columns)

    assert actual_columns == expected_columns, (
        f"Column mismatch!\nExpected: {expected_columns}\nActual: {actual_columns}"
    )
    print("+ Column order matches training exactly")

    # Predict
    model = artifact["model"]
    prediction = model.predict(X)[0]

    print(f"+ Prediction: {prediction} ({'Attrition' if prediction == 1 else 'No Attrition'})")

    # Verify prediction is valid
    assert prediction in (0, 1), f"Invalid prediction: {prediction}"

    # Get probability if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]
        print(f"+ Probability of attrition: {proba:.4f}")
        assert 0 <= proba <= 1, f"Invalid probability: {proba}"

    return prediction


def verify_feature_parity(artifact):
    """Verify that inference features match the artifact's expected columns."""
    print("\n" + "=" * 60)
    print("STEP 4: Verifying feature parity (inference vs artifact)...")
    print("=" * 60)

    # Create sample input and run through inference path
    sample = create_sample_input()
    parsed = parse_form_data(sample)
    X_infer = prep.preprocess_for_inference(parsed, artifact)

    # Verify columns match what's in the artifact
    expected_columns = artifact["feature_names"]
    actual_columns = list(X_infer.columns)

    assert actual_columns == expected_columns, (
        f"Column mismatch!\nExpected: {expected_columns}\nActual: {actual_columns}"
    )
    print("+ Inference feature columns match artifact feature_names exactly")

    # Also verify the model can predict on this
    model = artifact["model"]
    pred = model.predict(X_infer)[0]
    assert pred in (0, 1), f"Invalid prediction: {pred}"
    print(f"+ Model prediction on inference features: {pred}")

    return True


def test_api_validation(artifact):
    """Test the API validation logic with various inputs."""
    print("\n" + "=" * 60)
    print("STEP 5: Testing API validation logic...")
    print("=" * 60)

    # Import validation from app.py logic
    VALID_CATEGORIES = {
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        "Department": ["Research & Development", "Human Resources", "Sales"],
        "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        "Gender": ["Male", "Female"],
        "JobRole": [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ],
        "MaritalStatus": ["Married", "Single", "Divorced"],
        "OverTime": ["Yes", "No"],
    }

    VALID_RANGES = {
        "Age": (18, 60),
        "DailyRate": (102, 1499),
        "DistanceFromHome": (1, 29),
        "Education": (1, 5),
        "HourlyRate": (30, 100),
        "JobLevel": (1, 5),
        "MonthlyIncome": (1009, 19999),
        "NumCompaniesWorked": (0, 9),
        "PerformanceRating": (3, 4),
        "StockOptionLevel": (0, 3),
        "TotalWorkingYears": (0, 40),
        "TrainingTimesLastYear": (0, 6),
        "WorkLifeBalance": (1, 4),
        "YearsAtCompany": (0, 40),
        "YearsInCurrentRole": (0, 18),
        "YearsSinceLastPromotion": (0, 15),
        "YearsWithCurrManager": (0, 17),
        "EnvironmentSatisfaction": (1, 4),
        "JobInvolvement": (1, 4),
        "JobSatisfaction": (1, 4),
        "RelationshipSatisfaction": (1, 4),
    }

    def validate_input(data: dict) -> tuple[bool, str]:
        required_fields = list(VALID_CATEGORIES.keys()) + list(VALID_RANGES.keys())
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

        for field, valid_values in VALID_CATEGORIES.items():
            value = data.get(field)
            if value not in valid_values:
                return False, f"Invalid value for {field}: '{value}'. Must be one of: {valid_values}"

        for field, (min_val, max_val) in VALID_RANGES.items():
            try:
                value = int(data.get(field))
            except (ValueError, TypeError):
                return False, f"Invalid integer for {field}: '{data.get(field)}'"

            if value < min_val or value > max_val:
                return False, f"Value out of range for {field}: {value} (must be {min_val}-{max_val})"

        return True, ""

    # Test valid input
    sample = create_sample_input()
    parsed = parse_form_data(sample)
    is_valid, msg = validate_input(parsed)
    assert is_valid, f"Valid input rejected: {msg}"
    print("+ Valid input passes validation")

    # Test invalid categorical
    bad_input = parsed.copy()
    bad_input["BusinessTravel"] = "InvalidValue"
    is_valid, msg = validate_input(bad_input)
    assert not is_valid, "Invalid categorical should be rejected"
    print("+ Invalid categorical correctly rejected")

    # Test out of range
    bad_input = parsed.copy()
    bad_input["Age"] = 999
    is_valid, msg = validate_input(bad_input)
    assert not is_valid, "Out of range should be rejected"
    print("+ Out of range correctly rejected")

    # Test missing field
    bad_input = parsed.copy()
    del bad_input["Age"]
    is_valid, msg = validate_input(bad_input)
    assert not is_valid, "Missing field should be rejected"
    print("+ Missing field correctly rejected")

    return True


def main():
    """Run all smoke tests."""
    print("\n" + "#" * 60)
    print("# EMPLOYEE ATTRITION PREDICTION - SMOKE TEST")
    print("#" * 60)

    try:
        # Step 1: Train (or use existing)
        artifact = train_model()

        # Step 2: Load artifact
        artifact = load_artifact()

        # Step 3: Test inference
        prediction = test_inference(artifact)

        # Step 4: Verify feature parity
        verify_feature_parity(artifact)

        # Step 5: Test validation logic
        test_api_validation(artifact)

        print("\n" + "=" * 60)
        print("+ ALL SMOKE TESTS PASSED")
        print("=" * 60)
        print(f"Final prediction: {prediction} ({'Attrition' if prediction == 1 else 'No Attrition'})")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("X SMOKE TEST FAILED")
        print("=" * 60)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
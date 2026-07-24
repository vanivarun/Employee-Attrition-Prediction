"""
Employee Attrition Prediction - Flask Web Application

Uses shared preprocessing module for consistent feature engineering
between training and serving.

Security features:
- debug=False by default (set FLASK_DEBUG=true to enable)
- Server-side input validation (not trusting HTML min/max alone)
- No stack traces exposed in production
- Rate limiting on prediction endpoint
"""

import numpy as np
import os
import pickle
from functools import wraps
from time import time
from flask import Flask, request, jsonify, render_template

import preprocessing as prep

# Optional: SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # for type hints


def compute_shap_explanation(model, X, artifact):
    """Compute SHAP values for a single prediction."""
    if not SHAP_AVAILABLE:
        return None
    try:
        # Create background data for LinearExplainer (use zeros for interventional)
        background = np.zeros((1, X.shape[1]))
        explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)

        # For binary classification, get class 1 (attrition) values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = shap_values[0]  # single row
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]

        # Map encoded column names to readable names
        feature_map = {}
        thresholds = artifact.get("thresholds", prep.THRESHOLDS)
        cat_features = artifact["cat_feature_names"]
        num_features = artifact["numerical_columns"]

        for cat_name in cat_features:
            if "_" in cat_name:
                base = cat_name.split("_")[0]
                value = "_".join(cat_name.split("_")[1:])
                feature_map[cat_name] = f"{base}={value}"
            else:
                feature_map[cat_name] = cat_name

        for num_name in num_features:
            if num_name.endswith("_bool"):
                base = num_name.replace("_bool", "")
                if base == "Total_Satisfaction":
                    feature_map[num_name] = f"{base} >= {thresholds.get('Total_Satisfaction', 2.8)}"
                elif base == "Age":
                    feature_map[num_name] = f"{base} < {thresholds.get('Age', 35)}"
                elif base == "DailyRate":
                    feature_map[num_name] = f"{base} < {thresholds.get('DailyRate', 800)}"
                elif base == "Department":
                    feature_map[num_name] = f"{base} == R&D"
                elif base == "DistanceFromHome":
                    feature_map[num_name] = f"{base} > {thresholds.get('DistanceFromHome', 10)}"
                elif base == "JobRole":
                    feature_map[num_name] = f"{base} == Lab Tech"
                elif base == "HourlyRate":
                    feature_map[num_name] = f"{base} < {thresholds.get('HourlyRate', 65)}"
                elif base == "MonthlyIncome":
                    feature_map[num_name] = f"{base} < {thresholds.get('MonthlyIncome', 4000)}"
                elif base == "NumCompaniesWorked":
                    feature_map[num_name] = f"{base} > {thresholds.get('NumCompaniesWorked', 3)}"
                elif base == "TotalWorkingYears":
                    feature_map[num_name] = f"{base} < {thresholds.get('TotalWorkingYears', 8)}"
                elif base == "YearsAtCompany":
                    feature_map[num_name] = f"{base} < {thresholds.get('YearsAtCompany', 3)}"
                elif base == "YearsInCurrentRole":
                    feature_map[num_name] = f"{base} < {thresholds.get('YearsInCurrentRole', 3)}"
                elif base == "YearsSinceLastPromotion":
                    feature_map[num_name] = f"{base} < {thresholds.get('YearsSinceLastPromotion', 1)}"
                elif base == "YearsWithCurrManager":
                    feature_map[num_name] = f"{base} < {thresholds.get('YearsWithCurrManager', 1)}"
                else:
                    feature_map[num_name] = num_name
            else:
                feature_map[num_name] = num_name

        readable_names = [feature_map.get(col, col) for col in X.columns]

        # Get top contributing features
        contributions = list(zip(readable_names, shap_values, X.iloc[0].values))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_factors = []
        for name, shap_val, feat_val in contributions[:5]:
            direction = "increases" if shap_val > 0 else "decreases"
            impact = "attrition risk" if shap_val > 0 else "retention"
            top_factors.append(f"{name} ({direction} {impact})")

        return "; ".join(top_factors)
    except Exception:
        return None


app = Flask(__name__)

# Security: disable debug mode in production
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
app.config["DEBUG"] = DEBUG_MODE

# Rate limiting: simple in-memory store (use Redis in production)
_rate_limit_store = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # per window


def rate_limit(f):
    """Simple rate limiter decorator."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not DEBUG_MODE:
            client_ip = request.remote_addr
            now = time()
            # Clean old entries
            if client_ip in _rate_limit_store:
                _rate_limit_store[client_ip] = [
                    t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
                ]
            else:
                _rate_limit_store[client_ip] = []

            if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

            _rate_limit_store[client_ip].append(now)
        return f(*args, **kwargs)
    return decorated


# Load model artifact at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
print(f"Loading model from {MODEL_PATH}...")
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
encoder = artifact["encoder"]
feature_names = artifact["feature_names"]
numerical_columns = artifact["numerical_columns"]
categorical_columns = artifact["categorical_columns"]
thresholds = artifact.get("thresholds", prep.THRESHOLDS)

print(f"Model loaded: {type(model).__name__}")
print(f"Categorical features: {len(categorical_columns)}")
print(f"Numerical features: {len(numerical_columns)}")


# Valid values for categorical fields (from training data)
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

# Valid ranges for numerical fields (from training data)
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
    """
    Validate input data against known categories and ranges.
    Returns (is_valid, error_message).
    """
    # Check required fields
    required_fields = list(VALID_CATEGORIES.keys()) + list(VALID_RANGES.keys())
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate categorical fields
    for field, valid_values in VALID_CATEGORIES.items():
        value = data.get(field)
        if value not in valid_values:
            return False, f"Invalid value for {field}: '{value}'. Must be one of: {valid_values}"

    # Validate numerical fields
    for field, (min_val, max_val) in VALID_RANGES.items():
        try:
            value = int(data.get(field))
        except (ValueError, TypeError):
            return False, f"Invalid integer for {field}: '{data.get(field)}'"

        if value < min_val or value > max_val:
            return False, f"Value out of range for {field}: {value} (must be {min_val}-{max_val})"

    return True, ""


def parse_form_data(form) -> dict:
    """Parse form data with proper type conversion and field name mapping."""
    # Map form field names to internal names
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
        input_data[internal_name] = form.get(form_name, "")

    # Numerical fields (form name -> internal name)
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
            input_data[internal_name] = int(form.get(form_name, 0))
        except (ValueError, TypeError):
            input_data[internal_name] = 0

    return input_data


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
@rate_limit
def predict():
    """
    Handle prediction request from HTML form.
    Uses shared preprocessing for exact feature parity with training.
    """
    if request.method == "GET":
        return render_template("index.html")

    try:
        # Parse form data
        input_data = parse_form_data(request.form)

        # Server-side validation
        is_valid, error_msg = validate_input(input_data)
        if not is_valid:
            return render_template("index.html", prediction_text=f"Validation error: {error_msg}"), 400

    except Exception as e:
        # Generic error message in production
        error_msg = "Invalid input. Please check all fields are filled correctly."
        if DEBUG_MODE:
            error_msg += f" Details: {str(e)}"
        return render_template("index.html", prediction_text=error_msg), 400

    try:
        # Use shared preprocessing for exact feature parity
        X = prep.preprocess_for_inference(input_data, artifact)

        # Predict
        prediction = model.predict(X)[0]
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X)[0][1]

        if prediction == 0:
            result_text = "Employee Might Not Leave The Job"
            if probability is not None:
                result_text += f" (Confidence: {1 - probability:.1%})"
        else:
            result_text = "Employee Might Leave The Job"
            if probability is not None:
                result_text += f" (Confidence: {probability:.1%})"

        # Add SHAP explanation if available
        shap_explanation = compute_shap_explanation(model, X, artifact)
        if shap_explanation:
            result_text += f" | Top factors: {shap_explanation}"

        return render_template("index.html", prediction_text=result_text)

    except Exception as e:
        # No stack traces in production
        error_msg = "Prediction error. Please try again."
        if DEBUG_MODE:
            error_msg += f" Details: {str(e)}"
        return render_template("index.html", prediction_text=error_msg), 500


@app.route("/predict_api", methods=["POST"])
@rate_limit
def predict_api():
    """
    JSON API endpoint for programmatic access.
    Expects JSON with internal field names (matching training data).
    """
    try:
        input_data = request.get_json(force=True)

        if input_data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        # Server-side validation
        is_valid, error_msg = validate_input(input_data)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        # Use shared preprocessing
        X = prep.preprocess_for_inference(input_data, artifact)

        prediction = int(model.predict(X)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][1])

        # Add SHAP explanation if available
        shap_explanation = compute_shap_explanation(model, X, artifact)

        response = {
            "prediction": prediction,
            "probability": probability,
            "attrition_risk": "High" if prediction == 1 else "Low"
        }
        if shap_explanation:
            response["shap_explanation"] = shap_explanation

        return jsonify(response)

    except Exception as e:
        error_msg = "Prediction error"
        if DEBUG_MODE:
            error_msg += f": {str(e)}"
        return jsonify({"error": error_msg}), 400


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": type(model).__name__,
        "features": len(feature_names)
    })


if __name__ == "__main__":
    # Don't run with debug=True in production!
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG_MODE)
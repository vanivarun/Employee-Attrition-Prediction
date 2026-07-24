"""
SHAP explainability for Employee Attrition Prediction.

Provides lightweight SHAP value computation for single predictions
using the trained LogisticRegression model, plus optional plot generation.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import io
import base64


def get_feature_names_for_shap(artifact):
    """
    Get human-readable feature names for SHAP explanations.
    Maps encoded column names back to original feature concepts.
    """
    cat_feature_names = artifact["cat_feature_names"]
    numerical_columns = artifact["numerical_columns"]

    # Build mapping from encoded column -> original feature
    feature_map = {}

    # Categorical features: parse the one-hot encoded names
    for cat_name in cat_feature_names:
        if "_" in cat_name:
            base_feature = cat_name.split("_")[0]
            value = "_".join(cat_name.split("_")[1:])
            feature_map[cat_name] = f"{base_feature}={value}"
        else:
            feature_map[cat_name] = cat_name

    # Numerical features: use threshold descriptions
    thresholds = artifact.get("thresholds", {})
    for num_name in numerical_columns:
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
                feature_map[num_name] = f"{base} == Lab Technician"
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

    return feature_map


def compute_shap_values(model, X, artifact):
    """
    Compute SHAP values for a single-row DataFrame X using the trained model.

    Returns dict with:
    - shap_values: array of SHAP values for each feature
    - base_value: expected model output (log-odds)
    - feature_names: human-readable feature names
    - top_factors: list of top contributing features with direction
    """
    # Create SHAP explainer for linear model
    # For LogisticRegression, use LinearExplainer
    explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values shape is (1, n_features) for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (attrition)

    shap_values = shap_values[0]  # single row
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]  # class 1

    # Get human-readable feature names
    feature_map = get_feature_names_for_shap(artifact)
    readable_names = [feature_map.get(col, col) for col in X.columns]

    # Get top contributing features (by absolute SHAP value)
    contributions = list(zip(readable_names, shap_values, X.iloc[0].values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Format top factors with direction
    top_factors = []
    for name, shap_val, feat_val in contributions[:5]:
        direction = "increases" if shap_val > 0 else "decreases"
        top_factors.append({
            "feature": name,
            "shap_value": float(shap_val),
            "feature_value": float(feat_val),
            "direction": direction,
            "impact": "attrition risk" if shap_val > 0 else "retention"
        })

    return {
        "shap_values": {readable_names[i]: float(shap_values[i]) for i in range(len(shap_values))},
        "base_value": float(base_value),
        "top_factors": top_factors
    }


def format_explanation(shap_result):
    """
    Format SHAP result as human-readable string for API response.
    """
    factors = shap_result["top_factors"]
    lines = []
    for f in factors:
        feat_name = f["feature"]
        direction = f["direction"]
        impact = f["impact"]
        lines.append(f"{feat_name} ({direction} {impact})")
    return "; ".join(lines)


def generate_shap_bar_plot(shap_result, artifact, top_n=10):
    """
    Generate a horizontal bar plot of top SHAP values.
    Returns base64-encoded PNG string.
    """
    shap_vals = shap_result["shap_values"]
    features = list(shap_vals.keys())
    values = list(shap_vals.values())

    # Sort by absolute value
    sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_pairs[:top_n]]
    values = [x[1] for x in sorted_pairs[:top_n]]

    colors = ['#dc2626' if v > 0 else '#059669' for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP Value (impact on log-odds)')
    ax.set_title('Top Factors Driving Attrition Prediction', fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_shap_waterfall_plot(shap_result, artifact, top_n=10):
    """
    Generate a waterfall plot showing how each feature pushes from base value to prediction.
    Returns base64-encoded PNG string.
    """
    shap_vals = shap_result["shap_values"]
    base_value = shap_result["base_value"]

    # Sort by absolute value
    sorted_items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_items[:top_n]]
    values = [x[1] for x in sorted_items[:top_n]]

    # Calculate cumulative values for waterfall
    cumulative = base_value
    waterfall_values = []
    for v in values:
        cumulative += v
        waterfall_values.append(cumulative)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#dc2626' if v > 0 else '#059669' for v in values]

    # Plot bars
    y_pos = np.arange(len(features))
    lefts = [base_value] + waterfall_values[:-1]
    ax.barh(y_pos, values, left=lefts, color=colors, edgecolor='white', height=0.7)

    # Add base value line
    ax.axvline(x=base_value, color='gray', linestyle='--', alpha=0.5, label=f'Base value: {base_value:.3f}')
    ax.axvline(x=waterfall_values[-1], color='black', linestyle='--', alpha=0.7, label=f'Final: {waterfall_values[-1]:.3f}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Log-odds')
    ax.set_title('SHAP Waterfall: Base Value → Prediction', fontweight='bold', pad=15)
    ax.legend(loc='lower right')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def save_shap_plots(shap_result, artifact, output_dir="shap_plots"):
    """
    Save SHAP bar plot and waterfall plot to disk.
    Returns tuple of (bar_plot_path, waterfall_plot_path).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    bar_b64 = generate_shap_bar_plot(shap_result, artifact)
    waterfall_b64 = generate_shap_waterfall_plot(shap_result, artifact)

    bar_path = os.path.join(output_dir, "shap_bar_plot.png")
    waterfall_path = os.path.join(output_dir, "shap_waterfall_plot.png")

    with open(bar_path, "wb") as f:
        f.write(base64.b64decode(bar_b64))
    with open(waterfall_path, "wb") as f:
        f.write(base64.b64decode(waterfall_b64))

    return bar_path, waterfall_path


if __name__ == "__main__":
    # Quick test with sample data
    import pickle
    import preprocessing as prep

    with open("model.pkl", "rb") as f:
        artifact = pickle.load(f)

    sample = {
        "Age": 30, "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
        "Department": "Research & Development", "DistanceFromHome": 5,
        "Education": 3, "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3, "Gender": "Male",
        "HourlyRate": 65, "JobInvolvement": 3, "JobLevel": 2,
        "JobRole": "Laboratory Technician", "JobSatisfaction": 3,
        "MaritalStatus": "Single", "MonthlyIncome": 5000,
        "NumCompaniesWorked": 2, "OverTime": "Yes",
        "PerformanceRating": 3, "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1, "TotalWorkingYears": 5,
        "TrainingTimesLastYear": 2, "WorkLifeBalance": 3,
        "YearsAtCompany": 3, "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 2,
    }

    # Parse form data like app.py does
    field_mapping = {
        "BusinessTravel": "BusinessTravel", "Department": "Department",
        "Education Field": "EducationField", "Gender": "Gender",
        "Job Role": "JobRole", "Marital Status": "MaritalStatus",
        "Over Time": "OverTime",
    }
    num_fields = {
        "Age": "Age", "Daily Rate": "DailyRate", "Distance From Home": "DistanceFromHome",
        "Education": "Education", "Hourly Rate": "HourlyRate", "Job Level": "JobLevel",
        "Monthly Income": "MonthlyIncome", "Number of Companies Worked in": "NumCompaniesWorked",
        "Performance Rating": "PerformanceRating", "Stock Option Level": "StockOptionLevel",
        "Total Working Years": "TotalWorkingYears", "Training Times Last Year": "TrainingTimesLastYear",
        "Work Life Balance": "WorkLifeBalance", "Years At Company": "YearsAtCompany",
        "Years In Current Role": "YearsInCurrentRole", "Years Since Last Promotion": "YearsSinceLastPromotion",
        "Years With Curr Manager": "YearsWithCurrManager",
        "Environment Satisfaction": "EnvironmentSatisfaction", "Job Involvement": "JobInvolvement",
        "Job Satisfaction": "JobSatisfaction", "Relationship Satisfaction": "RelationshipSatisfaction",
    }

    internal = {}
    for form_name, internal_name in field_mapping.items():
        internal[internal_name] = sample.get(form_name, "")
    for form_name, internal_name in num_fields.items():
        internal[internal_name] = int(sample.get(form_name, 0))

    X = prep.preprocess_for_inference(internal, artifact)
    model = artifact["model"]

    result = compute_shap_values(model, X, artifact)
    print("Base value (log-odds):", result["base_value"])
    print("\nTop factors:")
    for f in result["top_factors"]:
        print(f"  {f['feature']}: SHAP={f['shap_value']:.4f} ({f['direction']} {f['impact']})")
    print("\nFormatted:", format_explanation(result))
"""
Employee Attrition Prediction - Model Training Pipeline

Uses shared preprocessing module for consistent feature engineering.
Evaluates multiple models with cross-validation and saves the best one
as a combined artifact (model + encoder + feature metadata).

Key principle: NO DATA LEAKAGE - encoder fitted ONLY on training data.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

# Optional imports for XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    LGBMClassifier = None

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

import preprocessing as prep

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data() -> pd.DataFrame:
    """Load the raw CSV data."""
    DATA_PATH = Path("Data") / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(DATA_PATH)
    return df


def build_pipelines():
    """Create dictionary of pipelines covering each model and imbalance strategy."""
    pipelines = {}

    # Logistic Regression variants
    pipelines["LogReg_baseline"] = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    pipelines["LogReg_class_weight"] = LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"
    )
    pipelines["LogReg_SMOTE"] = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])

    # Random Forest variants
    pipelines["RF_baseline"] = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200)
    pipelines["RF_class_weight"] = RandomForestClassifier(
        random_state=RANDOM_STATE, n_estimators=200, class_weight="balanced"
    )
    pipelines["RF_SMOTE"] = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200))
    ])

    # CatBoost variants
    pipelines["CatBoost_baseline"] = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False)
    pipelines["CatBoost_class_weight"] = CatBoostClassifier(
        random_state=RANDOM_STATE, verbose=False, auto_class_weights="Balanced"
    )
    pipelines["CatBoost_SMOTE"] = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", CatBoostClassifier(random_state=RANDOM_STATE, verbose=False))
    ])

    # XGBoost variants (if available)
    if HAS_XGB:
        pipelines["XGBoost_baseline"] = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_estimators=200
        )
        pipelines["XGBoost_scale_pos_weight"] = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_estimators=200,
            scale_pos_weight=5  # approximate ratio of neg:pos classes
        )
        pipelines["XGBoost_SMOTE"] = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_estimators=200))
        ])

    # LightGBM variants (if available)
    if HAS_LGB:
        pipelines["LGBM_baseline"] = LGBMClassifier(
            random_state=RANDOM_STATE, verbosity=-1, n_estimators=200
        )
        pipelines["LGBM_class_weight"] = LGBMClassifier(
            random_state=RANDOM_STATE, verbosity=-1, n_estimators=200, class_weight="balanced"
        )
        pipelines["LGBM_SMOTE"] = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_estimators=200))
        ])

    return pipelines


def tune_logreg(X_train, y_train):
    """
    Hyperparameter tuning for LogisticRegression using GridSearchCV.
    Returns best estimator with optimized C and penalty.
    """
    print("\n   Tuning LogisticRegression hyperparameters...")
    param_grid = {
        "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "penalty": ["l2"],  # l1 requires liblinear solver
        "solver": ["lbfgs", "saga"],
        "class_weight": [None, "balanced"],
        "max_iter": [2000],
    }

    base_lr = LogisticRegression(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        base_lr, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)

    print(f"   Best LogReg params: {grid.best_params_}")
    print(f"   Best CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate_pipelines(pipelines, X_train, y_train, X_test, y_test):
    """
    Run 5-fold stratified CV for each pipeline, rank by F1 then ROC-AUC,
    and return the best fitted model with results.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
            return_train_score=False,
            n_jobs=-1
        )
        results.append({
            "name": name,
            "accuracy": np.mean(scores["test_accuracy"]),
            "roc_auc": np.mean(scores["test_roc_auc"]),
            "f1": np.mean(scores["test_f1"])
        })

    # Sort by F1 then ROC-AUC (both descending)
    results.sort(key=lambda r: (r["f1"], r["roc_auc"]), reverse=True)

    best_name = results[0]["name"]
    best_pipe = pipelines[best_name]
    best_pipe.fit(X_train, y_train)

    # Test set evaluation
    y_pred = best_pipe.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1])
                    if hasattr(best_pipe, "predict_proba") else "N/A",
        "f1": f1_score(y_test, y_pred)
    }

    return best_pipe, results, best_name, test_metrics


def save_artifact(
    model,
    encoder,
    feature_names,
    numerical_columns,
    categorical_columns,
    cat_feature_names,
):
    """Save combined model artifact with all preprocessing metadata."""
    artifact = {
        "model": model,
        "encoder": encoder,
        "feature_names": feature_names,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "cat_feature_names": cat_feature_names,
        "thresholds": prep.THRESHOLDS,
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print("Combined artifact saved to model.pkl")


def main():
    print("=" * 60)
    print("Employee Attrition Prediction - Model Training")
    print("=" * 60)

    # MLflow setup
    if HAS_MLFLOW:
        mlflow.set_experiment("Employee_Attrition_Prediction")
        print("\nMLflow tracking enabled")
    else:
        print("\nMLflow not installed (pip install mlflow) - skipping experiment tracking")

    # 1. Load raw data
    print("\n1. Loading raw data...")
    df = load_data()
    print(f"   Raw data shape: {df.shape}")

    # 2. Train/test split FIRST (before any fitting to avoid leakage)
    print("\n2. Train/test split (80/20, stratified)...")
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["Attrition"], random_state=RANDOM_STATE
    )
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"   Train target distribution: {train_df['Attrition'].value_counts().to_dict()}")

    # 3. Preprocess training data using shared module (fits encoder on TRAIN only)
    print("\n3. Preprocessing training data (encoder fit on train only)...")
    X_train, y_train, encoder, feature_names, numerical_columns, categorical_columns, cat_feature_names = (
        prep.preprocess_training_pipeline(train_df, categorical_cols=prep.CATEGORICAL_COLS)
    )
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Feature names: {len(feature_names)} total")
    print(f"   - Categorical (encoded): {len(cat_feature_names)}")
    print(f"   - Numerical: {len(numerical_columns)}")

    # 4. Preprocess test data using SAME fitted encoder (no refitting!)
    print("\n4. Preprocessing test data (using fitted encoder)...")
    test_df_engineered = prep.build_features(test_df.copy())
    X_test_cat, X_test_num, y_test = prep.split_features_target(test_df_engineered)

    # Ensure categorical columns exist and are in correct order
    for col in categorical_columns:
        if col not in X_test_cat.columns:
            X_test_cat[col] = pd.Series([0] * len(test_df), index=test_df.index, dtype="category")
    X_test_cat = X_test_cat[categorical_columns]

    # Ensure numerical columns exist and are in correct order
    for col in numerical_columns:
        if col not in X_test_num.columns:
            X_test_num[col] = 0
    X_test_num = X_test_num[numerical_columns]

    # Transform using FITTED encoder
    X_test_cat_encoded = encoder.transform(X_test_cat)
    X_test_cat_df = pd.DataFrame(X_test_cat_encoded, columns=cat_feature_names, index=test_df.index)
    X_test_num.columns = X_test_num.columns.astype(str)

    X_test = pd.concat([X_test_cat_df, X_test_num], axis=1)
    expected_columns = cat_feature_names + numerical_columns
    X_test = X_test.reindex(columns=expected_columns, fill_value=0)

    print(f"   Test features shape: {X_test.shape}")

    # 5. Build pipelines
    print("\n5. Building model pipelines...")
    pipelines = build_pipelines()
    print(f"   {len(pipelines)} pipelines to evaluate")
    if HAS_XGB:
        print("   XGBoost: available")
    else:
        print("   XGBoost: not installed (pip install xgboost)")
    if HAS_LGB:
        print("   LightGBM: available")
    else:
        print("   LightGBM: not installed (pip install lightgbm)")

    # 6. Hyperparameter tuning for LogisticRegression (current best model)
    print("\n6. Hyperparameter tuning for LogisticRegression...")
    tuned_lr = tune_logreg(X_train, y_train)
    pipelines["LogReg_tuned"] = tuned_lr

    # 7. Evaluate all pipelines with cross-validation
    print("\n7. Cross-validating all pipelines (5-fold stratified)...")
    best_model, cv_results, best_name, test_metrics = evaluate_pipelines(
        pipelines, X_train, y_train, X_test, y_test
    )

    # MLflow logging
    if HAS_MLFLOW:
        with mlflow.start_run(run_name=f"train_{best_name}") as run:
            # Log parameters
            mlflow.log_param("best_model", best_name)
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("n_train", X_train.shape[0])
            mlflow.log_param("n_test", X_test.shape[0])
            mlflow.log_param("class_balance_train", y_train.value_counts().to_dict())
            mlflow.log_param("class_balance_test", y_test.value_counts().to_dict())

            # Log CV metrics for all models
            for r in cv_results:
                prefix = r['name'].replace(' ', '_').replace('-', '_')
                mlflow.log_metric(f"cv_{prefix}_f1", r['f1'])
                mlflow.log_metric(f"cv_{prefix}_roc_auc", r['roc_auc'])
                mlflow.log_metric(f"cv_{prefix}_accuracy", r['accuracy'])

            # Log test metrics for best model
            mlflow.log_metric("test_f1", test_metrics['f1'])
            mlflow.log_metric("test_roc_auc", test_metrics['roc_auc'])
            mlflow.log_metric("test_accuracy", test_metrics['accuracy'])

            # Log model artifact
            mlflow.sklearn.log_model(best_model, "model")

            # Log preprocessing artifact
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                save_artifact(
                    best_model, encoder, feature_names,
                    numerical_columns, categorical_columns, cat_feature_names
                )
                mlflow.log_artifact("model.pkl", "preprocessing")

            print(f"   MLflow run: {run.info.run_id}")

    # 8. Print results
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS (mean across 5 folds)")
    print("=" * 60)
    print(f"{'Model':<25} {'F1':>8} {'ROC-AUC':>8} {'Accuracy':>8}")
    print("-" * 60)
    for r in cv_results:
        print(f"{r['name']:<25} {r['f1']:.4f}  {r['roc_auc']:.4f}  {r['accuracy']:.4f}")

    print("-" * 60)
    print(f"\nBest pipeline (selected by F1, then ROC-AUC): {best_name}")
    print(f"Test set metrics:")
    print(f"  F1-score : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {test_metrics['roc_auc']:.4f}")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")

    # 9. Save combined artifact
    print("\n8. Saving combined model artifact...")
    save_artifact(
        best_model, encoder, feature_names,
        numerical_columns, categorical_columns, cat_feature_names
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
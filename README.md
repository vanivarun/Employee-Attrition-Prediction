# Employee Attrition Prediction using Machine Learning

![Employee Attrition](https://user-images.githubusercontent.com/53073185/87855289-e83cc300-c934-11ea-991f-59b7eb395eda.png)

Employee attrition is a critical challenge for organizations, leading to increased costs in hiring, training, and productivity loss.  
This project predicts employee attrition using Machine Learning techniques to help organizations take proactive retention measures.

---

## 📊 Dataset

- **IBM HR Analytics Employee Attrition & Performance**
- Source: [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Contains employee demographics, job role, compensation, satisfaction metrics, and attrition labels
- **1,470 records × 35 features**

---

## 🔍 Exploratory Data Analysis

- Analyzed categorical and numerical features
- Visualized attrition patterns across:
  - Job Role
  - Age
  - Monthly Income
  - Business Travel
  - Work-Life Balance
- Identified class imbalance (≈16% attrition rate)

---

## 🤖 Models Implemented & Evaluated

The following models were trained and compared using **5-fold stratified cross-validation**:

| Model | Variants |
|-------|----------|
| Logistic Regression | Baseline, Class Weight, SMOTE, **Tuned (GridSearchCV)** |
| Random Forest | Baseline, Class Weight, SMOTE |
| CatBoost | Baseline, Auto Class Weights, SMOTE |
| XGBoost* | Baseline, Scale Pos Weight, SMOTE |
| LightGBM* | Baseline, Class Weight, SMOTE |

**Total: up to 15 pipeline configurations evaluated** (XGBoost/LightGBM optional — auto-detected if installed)

---

## 📈 Model Evaluation Strategy

- **Train/Test Split**: 80/20 stratified
- **Cross-Validation**: 5-fold StratifiedKFold
- **Primary Metric**: **F1-Score** (handles class imbalance)
- **Tie-breaker**: **ROC-AUC**
- **Also tracked**: Accuracy

> **Note**: Since attrition is imbalanced (~16% positive class), F1 and ROC-AUC are more meaningful than raw accuracy.

---

## 🏆 Final Model Selection

The model is **automatically selected** based on cross-validated F1-score (then ROC-AUC) — not hardcoded.

The training script (`model.py`) evaluates all pipelines, runs **GridSearchCV on LogisticRegression** (current best family), and saves the best one as a single artifact (`model.pkl`) containing:
- The fitted model
- The fitted OneHotEncoder
- Feature names and column order
- Preprocessing thresholds (single source of truth)

This ensures **zero train/serve skew** — the Flask app uses the exact same preprocessing.

### 📊 Final Results (LogisticRegression with GridSearchCV)

| Metric | Value |
|--------|-------|
| **CV F1-Score** | **0.5618** |
| **CV ROC-AUC** | **0.8393** |
| **CV Accuracy** | **0.8895** |
| **Test F1-Score** | **0.4474** |
| **Test ROC-AUC** | **0.8037** |
| **Test Accuracy** | **0.8571** |

**Best LogisticRegression params**: `C=2.0, solver=lbfgs, max_iter=2000, penalty=l2, class_weight=None`

> **Why F1 dropped on test set**: The dataset is small (1,470 rows) with high class imbalance. The test set has only ~294 samples (≈47 positive cases), so test metrics have high variance. CV F1 (0.56) is the more reliable estimate.

---

## 📦 Model Persistence

- **Single canonical artifact**: `model.pkl` (pickle format)
- Contains: model + encoder + feature metadata + thresholds

---

## 🛠 Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn, CatBoost, Imbalanced-learn (SMOTE) |
| **Optional ML** | XGBoost, LightGBM (auto-detected) |
| **Explainability** | SHAP (LinearExplainer) |
| **Experiment Tracking** | MLflow |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web** | Flask, Jinja2, Bootstrap 5 |
| **Serialization** | Pickle (single combined artifact) |
| **Production** | Gunicorn, Docker |
| **CI/CD** | GitHub Actions |

---

## 🚀 Project Structure

```
Employee-Attrition-Prediction/
├── app.py                 # Flask web application
├── model.py               # Training pipeline (entry point)
├── preprocessing.py       # SHARED feature engineering (single source of truth)
├── test_pipeline.py       # End-to-end smoke test
├── test_preprocessing.py  # Unit tests for preprocessing
├── requirements.txt       # Pinned dependencies
├── model.pkl              # Combined model artifact (created by model.py)
├── Data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── templates/
│   └── index.html         # Web UI
├── static/
│   └── style.css          # Custom styles
├── explain.py             # SHAP explanation utilities
├── Dockerfile             # Container definition
├── .dockerignore          # Docker ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI pipeline
├── README.md
├── LICENSE
├── SECURITY.md
├── Procfile
└── .gitignore
```

---

## ⚙️ Installation

```bash
# Clone repo
git clone <your-repo-url>
cd Employee-Attrition-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Usage

### 1. Train the Model

```bash
python model.py
```

Outputs cross-validation comparison table and saves `model.pkl`.

### 2. Run the Web App

```bash
python app.py
```

Opens at `http://localhost:5000` — fill the form to get real-time attrition predictions with SHAP explanations.

### 3. API Endpoint

```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
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
    "YearsWithCurrManager": 2
  }'
```

Returns JSON:
```json
{
  "prediction": 0,
  "probability": 0.1140,
  "attrition_risk": "Low",
  "shap_explanation": "Department == R&D (decreases retention); Total_Satisfaction >= 2.8 (decreases retention); JobRole == Lab Tech (increases attrition risk); Age < 35 (increases attrition risk); OverTime=1 (increases attrition risk)"
}
```

### 4. Run Smoke Test

```bash
python test_pipeline.py
```

Verifies: training → artifact → inference → feature parity.

### 5. Run Unit Tests

```bash
python test_preprocessing.py
```

### 6. Docker

```bash
docker build -t attrition-pred .
docker run -p 5000:5000 attrition-pred
```

---

## 🔑 Key Design Decisions

| Concern | Solution |
|---------|----------|
| **Train/serve skew** | Single `preprocessing.py` module used by both `model.py` and `app.py` |
| **Data leakage** | OneHotEncoder fitted **only on training data** (after train/test split) |
| **Imbalance** | Evaluated via F1/ROC-AUC; pipelines include class_weight and SMOTE variants |
| **Reproducibility** | Fixed `random_state=42` everywhere |
| **Model selection** | Automatic (CV F1 → ROC-AUC), not hardcoded |
| **Artifact format** | Single pickle with model + encoder + metadata |
| **Security** | Debug=False by default, input validation, rate limiting, no stack traces in prod |
| **Explainability** | SHAP LinearExplainer for per-prediction feature attribution |

---

## 📝 Future Improvements

- [x] Hyperparameter tuning (GridSearchCV on LogisticRegression)
- [x] SHAP explainability for individual predictions
- [x] MLflow experiment tracking
- [x] Docker containerization
- [x] CI/CD pipeline (GitHub Actions)
- [x] More models: XGBoost, LightGBM (optional, auto-detected if installed)
- [ ] Calibration curves for probability reliability
- [ ] Unit tests for full pipeline

---

## 📄 License

MIT License — feel free to use and adapt.

---

## 🙏 Acknowledgments

- IBM HR Analytics dataset (Kaggle)
- CatBoost team for excellent gradient boosting library
- Scikit-learn and imbalanced-learn communities
- SHAP project for explainability
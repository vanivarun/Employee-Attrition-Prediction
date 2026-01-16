# Employee Attrition Prediction using Machine Learning

![Employee Attrition](https://user-images.githubusercontent.com/53073185/87855289-e83cc300-c934-11ea-991f-59b7eb395eda.png)

Employee attrition is a critical challenge for organizations, leading to increased costs in hiring, training, and productivity loss.  
This project focuses on predicting employee attrition using Machine Learning techniques to help organizations take proactive retention measures.

---

## 📊 Dataset
- **IBM HR Analytics Employee Attrition & Performance**
- Source: Kaggle  
- Link: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Dataset contains employee demographics, job role, compensation, satisfaction metrics, and attrition labels.

---

## 🔍 Exploratory Data Analysis
- Analyzed categorical and numerical features
- Visualized attrition patterns across:
  - Job Role
  - Age
  - Monthly Income
  - Business Travel
  - Work-Life Balance
- Identified class imbalance and key contributing factors

---

## 🤖 Models Implemented
The following machine learning models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Perceptron
- Stochastic Gradient Descent
- Decision Tree
- Random Forest
- Gradient Boosting Trees
- **CatBoost Classifier**

---

## 📈 Model Evaluation Strategy
- Train–Test Split (70% / 30%)
- K-Fold Cross Validation
- Accuracy used as evaluation metric
- Overfitting was explicitly checked by comparing training accuracy vs cross-validation accuracy

---

## 🏆 Final Model Selection
- Tree-based models like **Decision Tree** and **Random Forest** achieved very high training accuracy (≈100%) but showed **overfitting**
- **CatBoost Classifier** achieved the **highest cross-validation accuracy (~89%)**, indicating better generalization
- CatBoost demonstrated strong generalization performance and achieved the highest cross-validation accuracy among all evaluated models.

### ✅ Final Selected Model:
**CatBoost Classifier**

---

## 📦 Model Persistence
- The final trained model was saved using `joblib`
- File: `final_catboost_model.pkl`
- The saved model can be reused for prediction without retraining

---

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- CatBoost
- Jupyter Notebook

---

## 🚀 Future Scope
- Handle class imbalance using SMOTE
- Hyperparameter tuning
- Deploy as a REST API using Flask
- Build a web UI for HR teams

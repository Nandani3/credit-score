# Credit Score Prediction Model

This project uses machine learning to predict an individual's credit score category based on financial and personal data. It classifies individuals into categories such as **Poor**, **Standard**, and **Good** credit scores. The model is built using algorithms like **Random Forest** and **Logistic Regression**.

##  Project Objective

To build a machine learning model that can:

* Predict the credit score of users based on various input features.
* Help financial institutions make quick and informed decisions.
* Analyze which features contribute most to the credit score.

---

## Dataset

The dataset includes user financial details such as:

* Age
* Annual Income
* Monthly In-Hand Salary
* Number of Bank Accounts
* Number of Credit Cards
* Outstanding Debt
* Number of Loans
* Credit Mix
* Credit Utilization Ratio
* Credit History Age

You can use publicly available datasets like:

* [Kaggle Credit Score Dataset](https://www.kaggle.com/datasets)

---

##  Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Jupyter Notebook

---

## ⚙️ Model Workflow

1. **Data Cleaning**

   * Handling missing values
   * Encoding categorical data
   * Removing outliers

2. **Exploratory Data Analysis**

   * Feature correlation
   * Class distribution

3. **Feature Engineering**

   * Scaling numerical features
   * One-hot encoding categorical features

4. **Model Building**

   * Logistic Regression
   * Random Forest Classifier
   * Hyperparameter tuning

5. **Model Evaluation**

   * Accuracy
   * Precision, Recall, F1 Score
   * Confusion Matrix
   * ROC-AUC Score

---

##  Results

* **Random Forest Accuracy:** 87%
* **Logistic Regression Accuracy:** 82%
* The Random Forest performed better due to its ensemble nature and ability to handle feature importance more effectively.

---

##  Visualization

* Credit score distribution
* Feature importance plots
* Confusion matrix
* ROC curves

---

##  How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Nandani3/credit-score-model.git
   cd credit-score-model
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:

   ```bash
   jupyter notebook credit_score_model.ipynb
   ```

---

## 🧪 Troubleshooting

* **ImportError:** Make sure all required libraries (like `sklearn`, `matplotlib`, `seaborn`) are installed.
* **Data shape mismatch:** Recheck feature dimensions after encoding or scaling.
* **Model underfitting:** Try using ensemble methods like Random Forest or tune hyperparameters using GridSearchCV.

---

##  Future Enhancements

* Integrate with a frontend for user input and live predictions.
* Add more complex models like XGBoost or LightGBM.
* Deploy the model using Flask or Streamlit.
* Use SHAP values for detailed feature impact analysis.
* Introduce model interpretability dashboard (like LIME or SHAP).

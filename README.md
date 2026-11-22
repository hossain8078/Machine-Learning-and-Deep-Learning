# Machine Learning and Deep Learning

Coursework and projects from the Machine Learning and Deep Learning course (ANLT-202) in the Business Information Systems program at Saskatchewan Polytechnic, Regina. Each notebook runs in Google Colab. Click the "Open In Colab" badge at the top of any notebook.

**Author:** Md Alamgir Hossain, PMP, CSM, ITIL v5 · [LinkedIn](https://www.linkedin.com/in/alamgircanb) · [pmport.ca](https://pmport.ca)

---

## Featured: Capstone Project – Predicting Telecom Customer Churn

**Notebook:** [`Capstone_Project_MdAlamgir_Hossain.ipynb`](Capstone_Project_MdAlamgir_Hossain.ipynb)

An end-to-end churn prediction project on the IBM Telco Customer Churn dataset (7,043 customers, 21 columns). It compares two classical machine learning models with a deep learning model and ends with a business recommendation.

### What the notebook covers

1. **EDA and data cleaning.** It fixes `TotalCharges`, which was stored as text and had 11 blank values for new customers. It also identifies the class imbalance (73.5% stay, 26.5% churn) and visualizes the main churn drivers.
2. **Preprocessing pipeline.** A `ColumnTransformer` applies `StandardScaler` to numeric features and `OneHotEncoder` to categorical ones, on a stratified 80/20 train/test split.
3. **Classical models.** Logistic Regression and Random Forest, each tuned with `GridSearchCV` (5-fold CV, F1 scoring, class weighting).
4. **Deep learning.** A Keras neural network with two hidden ReLU layers, Dropout, a sigmoid output, class weights and early stopping. Seeds are fixed so the results are reproducible.
5. **Model comparison.** Accuracy, precision, recall, F1, ROC-AUC, training time and ROC curves.
6. **Business impact.** Decision-threshold tuning based on an estimated cost of missed churners and retention offers.
7. **Recommendation** and answers to eight conceptual questions.

### Results (test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.737 | 0.503 | **0.783** | 0.613 | **0.841** | **0.11 s** |
| Random Forest | **0.769** | **0.550** | 0.719 | **0.623** | 0.839 | 1.72 s |
| Neural Network | 0.744 | 0.512 | **0.783** | 0.619 | 0.838 | 19.62 s |

**Key findings**

- The strongest churn drivers are **month-to-month contracts, short tenure, fiber-optic internet, electronic-check payment and no tech support**.
- All three models perform almost the same (ROC-AUC within 0.003), so the choice comes down to cost, interpretability and complexity.
- With a cost-based threshold of 0.35 instead of 0.5, Logistic Regression catches **339 of 374 churners**. Under the stated assumptions, that cuts the estimated cost of churn by about **24%** compared with using no model.
- **Recommendation: Logistic Regression.** It matches the other models' performance, is fully interpretable through its coefficients and odds ratios, and is the simplest to deploy and maintain.

---

## Lab notebooks

| Notebook | Topic |
|---|---|
| [`Alamgir_BIS_Wine_Classification_EDA_and_RegressionModel.ipynb`](Alamgir_BIS_Wine_Classification_EDA_and_RegressionModel.ipynb) | Lab 1: multiclass classification on the Wine dataset (EDA, model training and evaluation) |
| [`Completed_Lab_Exercise_2_Diabetes_Regression.ipynb`](Completed_Lab_Exercise_2_Diabetes_Regression.ipynb) | Lab 2: linear regression on the Diabetes dataset (BMI-only vs. all-feature model) |
| [`Lab_4_Titanic_Preprocessing.ipynb`](Lab_4_Titanic_Preprocessing.ipynb) | Lab 4: preprocessing a real-world dataset (missing values, categorical encoding, pipeline classifier) |
| [`05_Pipelines.ipynb`](05_Pipelines.ipynb) | Automated machine learning pipelines and hyperparameter tuning |

## Tools and libraries

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · TensorFlow / Keras · Google Colab

## How to run

Open any notebook in Google Colab with its badge, then choose **Runtime → Run all**. The datasets load from public URLs or built-in scikit-learn datasets, so you don't need to download anything.

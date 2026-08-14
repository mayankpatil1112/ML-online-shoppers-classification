# Online Shopper Purchase Intention Prediction — ML Assignment 2

## 1. Problem Statement

The objective is to build and compare classification models that predict whether an online shopping session will end in a purchase. The target variable is `Revenue`.

This project implements the classification workflow required for the BITS Pilani Machine Learning Assignment 2 and provides an interactive Streamlit application for evaluating the trained models on test data.

## 2. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**UCI dataset ID:** 468  
**DOI:** 10.24432/C5F88Q

The dataset contains **12,330 online shopping sessions and 17 input features**, with `Revenue` as the binary target. UCI reports no missing values. The dataset contains numerical and categorical attributes such as page visits, page durations, bounce rate, exit rate, page value, month, operating system, browser, region, traffic type, visitor type and weekend status.

Dataset source:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

### Target

- `Revenue = 0`: session did not result in a purchase
- `Revenue = 1`: session resulted in a purchase

### Class Distribution

- No purchase: 10,422
- Purchase: 1,908

Because the positive class is smaller, class imbalance is considered in the Logistic Regression, Decision Tree and Random Forest models using `class_weight="balanced"`.

## 3. Data Preparation

1. Loaded the original UCI CSV.
2. Separated `Revenue` as the target.
3. Identified numerical and categorical features.
4. Applied `StandardScaler` to numerical variables.
5. Applied `OneHotEncoder(handle_unknown="ignore")` to categorical variables.
6. Used a stratified 80:20 train/test split.
7. Random state: 42.
8. Saved the fixed test set as `test_data.csv`.

All preprocessing is implemented inside scikit-learn pipelines to prevent data leakage between training and testing.

## 4. Models Used

The five models required by the assignment are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

## 5. Evaluation Metrics

Every model is evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 6. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8500 | 0.8962 | 0.5107 | 0.7487 | 0.6072 | 0.5330 |\n| Decision Tree | 0.8410 | 0.8850 | 0.4921 | 0.8168 | 0.6142 | 0.5483 |\n| kNN | 0.8767 | 0.8330 | 0.7191 | 0.3351 | 0.4571 | 0.4349 |\n| Naive Bayes | 0.6736 | 0.7932 | 0.2937 | 0.7880 | 0.4279 | 0.3234 |\n| Random Forest | 0.8978 | 0.9213 | 0.7444 | 0.5183 | 0.6111 | 0.5665 |\n

## 7. Observations

### Logistic Regression
Provides a strong baseline and identifies a substantial portion of purchasing sessions. Scaling and one-hot encoding are important because the dataset contains mixed feature types.

### Decision Tree
Provides the highest Recall and F1 among the five models on this fixed test split. It captures non-linear relationships but is less accurate overall than the Random Forest.

### kNN
Achieves good overall accuracy but has lower recall for the purchase class. Distance-based classification is sensitive to feature representation, so numerical scaling is used.

### Naive Bayes
Has the highest recall among the models except the Decision Tree, but its overall accuracy and F1 are lower. The independence assumption is restrictive for correlated shopping-session variables.

### Random Forest
Achieves the highest Accuracy, AUC and MCC on the fixed test split. The ensemble of trees captures non-linear interactions while reducing the variance of a single decision tree.

### Overall Winner

**Random Forest** is selected as the overall winner because it gives the best combination of Accuracy, AUC and MCC on the fixed test set. The Decision Tree has a marginally higher F1 score, so F1 should be highlighted separately when the priority is identifying purchase sessions.

## 8. Streamlit Application

The application includes:

- CSV test-data upload
- Model-selection dropdown
- Comparison table for all five models
- Accuracy
- AUC
- Precision
- Recall
- F1
- MCC
- Confusion matrix
- Classification report
- Test-data preview

The application evaluates the uploaded test data using the same saved pipelines used during model development.

## 9. GitHub Repository Link

**Replace with your actual repository URL:**

`https://github.com/<YOUR_GITHUB_USERNAME>/ML-online-shoppers-classification`

## 10. Live Streamlit App Link

**Replace after deployment:**

`https://<YOUR-APP-NAME>.streamlit.app/`

## 11. Project Structure

```text
ML-online-shoppers-classification/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
│
├── data/
│   └── online_shoppers_intention.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## 12. Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain the models:

```bash
python train_models.py
```

## 13. Deployment

The app is designed for Streamlit Community Cloud.

1. Push the complete repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the GitHub repository.
5. Select the `main` branch.
6. Select `app.py`.
7. Deploy.
8. Test CSV upload and every model.
9. Copy the generated `streamlit.app` URL into the final submission PDF.

## 14. Academic Integrity

This implementation is intended for learning support. Before submission, run the project yourself, inspect the code, understand the preprocessing and metrics, and customize the presentation where appropriate. The BITS assignment states that GitHub history and copied datasets/models/outputs may be checked.

## 15. Dataset Citation

Sakar, C. O., & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

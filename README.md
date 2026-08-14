# ML Assignment 2 — Online Shoppers Purchasing Intention Classification
Mayank Patil 2025ac05450
## a. Problem Statement
Predict whether an online shopping session will result in a purchase using multiple classification models and compare their performance through an interactive Streamlit application.

## b. Dataset Description
**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Instances:** 12,330  
**Input Features:** 17  
**Target:** `Revenue`  
**Problem:** Binary classification  
**0 / False:** No Purchase  
**1 / True:** Purchase  

The dataset satisfies the assignment requirement of at least 500 instances and at least 12 features.

## c. GitHub Repository Link
https://github.com/mayankpatil1112/ml-online-shoppers-classification

## Live Streamlit App
https://ml-online-shoppers-classification-hsqqpt9yimx6my3qsncfak.streamlit.app/

## d. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.84996 | 0.89620 | 0.70750 | 0.53272 | 0.60722 | 0.52648 |
| Decision Tree | 0.84104 | 0.88498 | 0.69424 | 0.55026 | 0.61417 | 0.52035 |
| kNN | 0.87672 | 0.83300 | 0.79167 | 0.31937 | 0.45714 | 0.42178 |
| Naive Bayes | 0.67356 | 0.79316 | 0.30612 | 0.70419 | 0.42786 | 0.28920 |
| Random Forest (Ensemble) | **0.89781** | **0.92133** | **0.74436** | 0.51832 | 0.61111 | **0.56650** |

## Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Strong baseline with good AUC and balanced overall performance. |
| Decision Tree | Captures non-linear relationships and achieves the highest F1 score among the five models. |
| kNN | Good accuracy, but comparatively lower recall and F1 for purchase prediction. |
| Naive Bayes | Higher recall, but substantially lower accuracy, precision, F1 and MCC. |
| Random Forest (Ensemble) | Highest Accuracy, AUC and MCC, giving the strongest overall performance. |

## Overall Winner
**Random Forest (Ensemble)** is the overall winner based on the strongest combined performance across Accuracy, AUC and MCC. Decision Tree has the highest F1 score, which is also acknowledged.

## Train/Test Split
- Training instances: 9,864
- Test instances: 2,466
- Test size: 20%
- Random state: 42
- Stratified split: Yes

## Streamlit Application Features
- CSV test-data upload
- Model selection dropdown
- Model comparison table
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Test-data preview

## Project Structure
```text
ML-online-shoppers-classification/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
├── data/
│   └── online_shoppers_intention.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed using Streamlit Community Cloud.

**Live App:** https://ml-online-shoppers-classification-hsqqpt9yimx6my3qsncfak.streamlit.app/

## Dataset Citation
Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository.

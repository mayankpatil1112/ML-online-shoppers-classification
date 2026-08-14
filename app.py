from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

st.set_page_config(page_title="Online Shopper Purchase Prediction", page_icon="🛒", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_TEST = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

st.title("🛒 Online Shopper Purchase Intention")
st.markdown(
    "### Machine Learning Classification Model Lab\n"
    "Predict whether an online shopping session is likely to end in a purchase."
)
st.caption("BITS Pilani M.Tech (AIML/DSE) — Machine Learning Assignment 2")

with st.sidebar:
    st.header("Model & Data")
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    selected_model = st.selectbox("Select classification model", list(MODEL_FILES))
    st.divider()
    st.write("**Target:** Revenue")
    st.write("False = No purchase")
    st.write("True = Purchase")

@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}

models = load_models()

try:
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        source = "Uploaded CSV"
    else:
        data = pd.read_csv(DEFAULT_TEST)
        source = "Bundled test_data.csv"
except Exception as e:
    st.error(f"Unable to read the CSV: {e}")
    st.stop()

if "Revenue" not in data.columns:
    st.error("The CSV must contain the Revenue target column for evaluation.")
    st.stop()

X = data.drop(columns=["Revenue"])
y = data["Revenue"].astype(int)

expected_features = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend"
]
missing = [c for c in expected_features if c not in X.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

if not set(y.unique()).issubset({0,1}):
    st.error("Revenue must be encoded as 0/1.")
    st.stop()

st.info(f"Evaluation data source: **{source}**")

c1,c2,c3,c4=st.columns(4)
c1.metric("Test Sessions", len(data))
c2.metric("Features", len(expected_features))
c3.metric("Purchases", int(y.sum()))
c4.metric("No Purchase", int((y==0).sum()))

def evaluate(model):
    pred=model.predict(X)
    proba=model.predict_proba(X)[:,1]
    return {
        "Accuracy":accuracy_score(y,pred),
        "AUC":roc_auc_score(y,proba),
        "Precision":precision_score(y,pred,zero_division=0),
        "Recall":recall_score(y,pred,zero_division=0),
        "F1":f1_score(y,pred,zero_division=0),
        "MCC":matthews_corrcoef(y,pred),
    },pred

rows=[]
preds={}
for name,model in models.items():
    m,p=evaluate(model)
    m["ML Model Name"]=name
    rows.append(m)
    preds[name]=p
comparison=pd.DataFrame(rows)[["ML Model Name","Accuracy","AUC","Precision","Recall","F1","MCC"]]

st.subheader("Model Comparison")
st.dataframe(comparison.style.format({c:"{:.4f}" for c in comparison.columns[1:]}),use_container_width=True,hide_index=True)

st.subheader(f"Selected Model — {selected_model}")
selected_metrics, selected_pred=evaluate(models[selected_model])
cols=st.columns(6)
for col,label in zip(cols,["Accuracy","AUC","Precision","Recall","F1","MCC"]):
    col.metric(label,f"{selected_metrics[label]:.4f}")

st.subheader("Confusion Matrix")
cm=confusion_matrix(y,selected_pred)
fig,ax=plt.subplots(figsize=(5,4))
im=ax.imshow(cm)
ax.set_xticks([0,1],["No Purchase","Purchase"])
ax.set_yticks([0,1],["No Purchase","Purchase"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(selected_model)
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center")
fig.colorbar(im,ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
report=pd.DataFrame({
    "Class":["No Purchase (0)","Purchase (1)"],
    "Precision":[precision_score(y,selected_pred,pos_label=0,zero_division=0),precision_score(y,selected_pred,pos_label=1,zero_division=0)],
    "Recall":[recall_score(y,selected_pred,pos_label=0,zero_division=0),recall_score(y,selected_pred,pos_label=1,zero_division=0)],
    "F1":[f1_score(y,selected_pred,pos_label=0,zero_division=0),f1_score(y,selected_pred,pos_label=1,zero_division=0)]
})
st.dataframe(report.style.format({"Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"}),use_container_width=True,hide_index=True)

with st.expander("View test data"):
    st.dataframe(data,use_container_width=True)

st.caption("Academic demonstration only. This model predicts session purchase intention and is not a financial or medical decision system.")

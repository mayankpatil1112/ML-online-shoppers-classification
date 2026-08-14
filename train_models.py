from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "online_shoppers_intention.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)

categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numeric_features = [c for c in X.columns if c not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", max_features="sqrt",
        random_state=42, n_jobs=-1
    ),
}

rows = []
for name, classifier in models.items():
    pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", classifier),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    filename = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "kNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
    }[name]
    joblib.dump(pipe, MODEL_DIR / f"{filename}.joblib", compress=3)

pd.DataFrame(rows).to_csv(BASE_DIR / "model_metrics.csv", index=False)
pd.concat([X_test, y_test.rename("Revenue")], axis=1).to_csv(BASE_DIR / "test_data.csv", index=False)

print(pd.DataFrame(rows).to_string(index=False))
print("\nTraining completed.")

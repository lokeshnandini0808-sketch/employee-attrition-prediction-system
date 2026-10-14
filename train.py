
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("employee_data.csv")

# Remove identifier and target from features
X = df.drop(columns=["Attrition", "EmployeeNumber", "EmployeeCount",
                     "Over18", "StandardHours"], errors="ignore")
y = df["Attrition"].map({"Yes": 1, "No": 0})

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0)
    })

    trained_models[name] = pipeline

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select by attrition-class F1 score
best_name = results_df.loc[
    results_df["F1"].idxmax(), "Model"
]

best_model = trained_models[best_name]

joblib.dump(best_model, "attrition_model_v2.pkl")

# Save feature list for the website
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("\nBest Model:", best_name)
print("New model saved as attrition_model_v2.pkl")

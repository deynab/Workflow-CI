import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

X_train = pd.read_csv('breast_cancer_preprocessing/X_train.csv')
X_test  = pd.read_csv('breast_cancer_preprocessing/X_test.csv')
y_train = pd.read_csv('breast_cancer_preprocessing/y_train.csv').squeeze()
y_test  = pd.read_csv('breast_cancer_preprocessing/y_test.csv').squeeze()

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm, cmap='Blues')
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    ax.set_title('Confusion Matrix - Breast Cancer')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()

    mlflow.sklearn.log_model(model, "model")

    with open("latest_run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("CI Training selesai!")
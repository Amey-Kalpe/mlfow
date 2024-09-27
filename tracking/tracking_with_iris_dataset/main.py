from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import mlflow

iris = pd.read_csv("iris.csv")

attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris.columns = attributes

# print(iris.head())
encoder = LabelEncoder()
scaler = StandardScaler()

X, y = scaler.fit_transform(iris.drop(columns=["class"])), encoder.fit_transform(
    iris["class"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLFlow set tracking URI for mlflow server
mlflow.set_tracking_uri("")

artifact_loc = Path.cwd().joinpath("iris_artifacts").as_uri()
exp = mlflow.get_experiment_by_name(name="iris_exp")

if exp:
    exp_id = exp.experiment_id
else:
    exp_id = mlflow.create_experiment(
        name="iris_exp",
        tags={"dataset": "iris", "model": "random_forest"},
        artifact_location=artifact_loc,
    )


with mlflow.start_run(experiment_id=exp_id):
    train_data = pd.DataFrame(
        X_train, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    train_data["class"] = y_train
    train_data.to_csv("data/train_data.csv")
    mlflow.log_artifact("data/train_data.csv")

    test_data = pd.DataFrame(
        X_test, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    test_data["class"] = y_test
    train_data.to_csv("data/test_data.csv")
    mlflow.log_artifact("data/test_data.csv")

    mlflow.sklearn.autolog(log_input_examples=True)
    n_estimators = 50
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    random_forest.fit(X_train, y_train)
    # mlflow.log_param("n_estimators", n_estimators)
    # mlflow.sklearn.log_model(random_forest, artifact_path="iris_models")

    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # mlflow.log_metric("accuracy", accuracy)

    precision = precision_score(y_test, y_pred, average="macro")
    print(f"Precision: {precision:.2f}")
    # mlflow.log_metric("precision", precision)

    recall = recall_score(y_test, y_pred, average="macro")
    print(f"Recall: {recall:.2f}")
    # mlflow.log_metric("recall", recall)

last_run = mlflow.last_active_run()
print(last_run.info.run_id)

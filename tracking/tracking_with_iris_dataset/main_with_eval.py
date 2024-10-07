from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.dummy import DummyClassifier
from mlflow.metrics import make_metric
from mlflow.models import MetricThreshold
import mlflow
import mlflow.sklearn
import joblib


def dump_model(model, model_path):
    joblib.dump(model, model_path)


iris = pd.read_csv("iris.csv")
iris_weight_map = {"Setosa": 1, "Versicolor": 1.5, "Virginica": 1.2}


attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "variety"]
iris.columns = attributes

iris["weight"] = iris["variety"].map(iris_weight_map)

encoder = LabelEncoder()
scaler = StandardScaler()

X, y = scaler.fit_transform(iris.drop(columns=["variety"])), encoder.fit_transform(
    iris["variety"]
)

X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(
    iris.drop(columns=["variety", "weight"]),
    iris["variety"],
    iris["weight"],
    test_size=0.2,
    random_state=42,
)

# MLFlow set tracking URI for mlflow server
mlflow.set_tracking_uri("")

# artifact_loc = Path.cwd().joinpath("iris_artifacts").as_uri()

exp_name = "iris_evaluation"
exp = mlflow.get_experiment_by_name(name=exp_name)

if exp:
    exp_id = exp.experiment_id
else:
    exp_id = mlflow.create_experiment(
        name=exp_name,
        tags={"dataset": "iris", "model": "random_forest"},
        # artifact_location=artifact_loc,
    )


def train_dummy_classifier(train_x, train_y, **kwargs):
    if "n_estimators" in kwargs:
        baseline_classifier = DummyClassifier(n_estimators=kwargs["n_estimators"])
    else:
        baseline_classifier = DummyClassifier()
    baseline_classifier.fit(train_x, train_y)
    return baseline_classifier


with mlflow.start_run(experiment_id=exp_id):
    pd.concat([X_train, y_train], axis=1).to_csv("data/train_data.csv")
    mlflow.log_artifact("data/train_data.csv")

    pd.concat([X_test, y_test], axis=1).to_csv("data/test_data.csv")
    mlflow.log_artifact("data/test_data.csv")

    mlflow.sklearn.autolog(log_input_examples=True)
    n_estimators = 50
    random_forest = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    random_forest.fit(X_train, y_train)
    # mlflow.log_param("n_estimators", n_estimators)
    mlflow.sklearn.log_model(random_forest, "iris_artifacts")

    def sklearn_accuracy_score(predictions, targets):
        return accuracy_score(predictions, targets)

    baseline_classifier = train_dummy_classifier(X_train, y_train)
    joblib.dump(baseline_classifier, "baseline_classifier.pkl")

    mlflow.sklearn.log_model(baseline_classifier, "baseline_classifier")

    artifact_uri = mlflow.get_artifact_uri("iris_artifacts")
    baseline_artifact_uri = mlflow.get_artifact_uri("baseline_classifier")

    sklearn_accuracy = make_metric(
        eval_fn=sklearn_accuracy_score,
        greater_is_better=True,
        name="sklearn_accuracy_score",
    )

    thresholds = {
        "sklearn_accuracy_score": MetricThreshold(
            threshold=0.9,
            min_absolute_change=0.05,
            min_relative_change=0.05,
            greater_is_better=True,
        )
    }

    mlflow.evaluate(
        artifact_uri,
        targets="variety",
        evaluators=["default"],
        data=pd.concat([X_test, y_test], axis=1),
        model_type="classifier",
        custom_metrics=[sklearn_accuracy],
        validation_thresholds=thresholds,
        baseline_model=baseline_artifact_uri,
    )

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

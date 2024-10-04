""" Start mlflow tracking server before running this module 
This module assumes sklearn is not supported by mlflow for
the purpose of implementing custom python model.
"""

import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature, infer_signature
from pathlib import Path
from predictor import predictor
import joblib
from mlflow.models import make_metric
from sklearn.dummy import DummyRegressor
from mlflow.models import MetricThreshold

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.05)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.1)
args = parser.parse_args()


# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_dummy_regressor(train_x, train_y, **kwargs):
    if "alpha" in kwargs and "l1_ratio" in kwargs:
        baseline_regressor = DummyRegressor(
            alpha=kwargs["alpha"], l1_ratio=kwargs["l1_ratio"]
        )
    else:
        baseline_regressor = DummyRegressor()
    baseline_regressor.fit(train_x, train_y)
    return baseline_regressor


def dump_model(model, model_path):
    joblib.dump(model, model_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="")
    mlflow.sklearn.autolog(
        log_input_examples=False, log_model_signatures=False, log_models=False
    )

    print("The set tracking uri is ", mlflow.get_tracking_uri())

    experiment_name = "exp_custom_model_evaluation"
    exp = mlflow.get_experiment_by_name(name=experiment_name)

    if exp:
        exp_id = exp.experiment_id
    else:
        exp_id = mlflow.create_experiment(
            name=experiment_name,
            tags={"version": "v1", "priority": "p1"},
            # artifact_location=Path.cwd().joinpath("myartifacts").as_uri(),
        )
    get_exp = mlflow.get_experiment(exp_id)

    print("Name: {}".format(get_exp.name))
    print("Experiment_id: {}".format(get_exp.experiment_id))
    print("Artifact Location: {}".format(get_exp.artifact_location))
    print("Tags: {}".format(get_exp.tags))
    print("Lifecycle_stage: {}".format(get_exp.lifecycle_stage))
    print("Creation timestamp: {}".format(get_exp.creation_time))

    with mlflow.start_run(experiment_id=exp_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        baseline_regressor = train_dummy_regressor(train_x, train_y)
        baseline_predicted_qualities = baseline_regressor.predict(test_x)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        sklearn_model_path = "sklearn_model.pkl"
        baseline_model_path = "baseline_sklearn_regressor.pkl"
        dump_model(lr, sklearn_model_path)
        dump_model(baseline_regressor, baseline_model_path)

        data_dir = "red-wine-data"
        train_x.to_csv(f"{data_dir}/train_data.csv")
        test_x.to_csv(f"{data_dir}/test_data.csv")

        sklearn_artifacts = {"sklearn_model": sklearn_model_path, "data": data_dir}
        baseline_artifacts = {
            "baseline_sklearn_regressor": baseline_model_path,
            "data": data_dir,
        }

        class SklearnModel(mlflow.pyfunc.PythonModel):
            def __init__(self, artifact_name):
                self.artifact_name = artifact_name

            def load_context(self, context):
                self.sklearn_model = joblib.load(context.artifacts[self.artifact_name])

            def predict(self, context, model_input):
                return self.sklearn_model.predict(model_input.values)

        thresholds = {
            "mean_squared_error": MetricThreshold(
                threshold=0.5,
                min_absolute_change=0.05,
                min_relative_change=0.05,
                greater_is_better=False,
            )
        }

        mlflow.pyfunc.log_model(
            artifact_path="custom_sklearn_artifacts",
            python_model=SklearnModel(sklearn_model_path.split(".")[0]),
            artifacts=sklearn_artifacts,
            code_paths=["main.py"],
            pip_requirements="../requirements.txt",
        )

        mlflow.pyfunc.log_model(
            artifact_path="baseline_sklearn_regressor_artifacts",
            python_model=SklearnModel(baseline_model_path.split(".")[0]),
            artifacts=baseline_artifacts,
            code_paths=["main.py"],
            pip_requirements="../requirements.txt",
        )

        # Create Custom Metrics
        # While creating custom metrics, use _builtin_metrics if not dealing with the mlflow builtin metrics and
        # use builtin_metrics otherwise, same goes with eval_df, see below
        def square_diff_plus_one(eval_df, _builtin_metrics):
            return np.sum(np.abs((eval_df["prediction"] - eval_df["target"] + 1) ** 2))

        def sum_on_target_by_two(_eval_df, builtin_metrics):
            return builtin_metrics["sum_on_target"] / 2

        metric_squared_diff_plus_one = make_metric(
            eval_fn=square_diff_plus_one,
            greater_is_better=False,
            name="squared_diff_plus_one",
        )

        metric_sum_on_target_by_two = make_metric(
            eval_fn=sum_on_target_by_two,
            greater_is_better=True,
            name="sum_on_target_by_two",
        )

        sklearn_artifacts_uri = mlflow.get_artifact_uri("custom_sklearn_artifacts")
        baseline_artifacts_uri = mlflow.get_artifact_uri(
            "baseline_sklearn_regressor_artifacts"
        )

        # Evaluate the model
        mlflow.evaluate(
            model=sklearn_artifacts_uri,
            data=test,
            model_type="regressor",
            targets="quality",
            evaluators=["default"],
            custom_metrics=[metric_squared_diff_plus_one, metric_sum_on_target_by_two],
            validation_thresholds=thresholds,
            baseline_model=baseline_artifacts_uri,
        )

        # Load the custom model
        # custom_model = mlflow.pyfunc.load_model(
        #     model_uri="runs:/76354552393f44d28faef66fed93a612/custom_sklearn_artifacts"
        # )
        # predicted_qualities = custom_model.predict(test_x)
        # print(predicted_qualities)
        # rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
        # print("  RMSE: %s" % rmse)
        # print("  MAE: %s" % mae)
        # print("  R2: %s" % r2)
        print(f"Current active run: {mlflow.active_run()}")

    print(f"Current active run: {mlflow.last_active_run()}")

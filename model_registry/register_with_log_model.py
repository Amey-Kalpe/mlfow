""" Start mlflow tracking server before running this module
"""

import os
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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()


def check_and_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("Directory created successfully.")
    else:
        print("Directory already exists.")


# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = pd.read_csv("../data/red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
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
            artifact_location=Path.cwd().joinpath("myartifacts").as_uri(),
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

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        data_dir = "red-wine-data"
        check_and_create_dir(data_dir)
        train_x.to_csv(f"{data_dir}/train_data.csv")
        test_x.to_csv(f"{data_dir}/test_data.csv")

        model_signature = infer_signature(test_x, predicted_qualities)

        # Register the model while logging it
        mlflow.sklearn.log_model(
            lr,
            artifact_path="myartifacts",
            signature=model_signature,
            registered_model_name="elasticnet-api",
        )

        mlflow.log_artifacts(data_dir)

        artifacts_uri = mlflow.get_artifact_uri("myartifacts")

        # Evaluate the model
        mlflow.evaluate(
            model=artifacts_uri,
            data=test,
            model_type="regressor",
            targets="quality",
            evaluators=["default"],
        )

        print(f"Current active run: {mlflow.active_run()}")

    print(f"Current active run: {mlflow.last_active_run()}")

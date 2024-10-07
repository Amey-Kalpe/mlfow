import mlflow
import pickle
import mlflow.sklearn

model_file = "elasticnet-regressor.pkl"
model = pickle.load(open(model_file, "rb"))

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "register_custom_model"

exp = mlflow.get_experiment_by_name(experiment_name)
if exp:
    exp_id = exp.experiment_id
else:
    exp_id = mlflow.create_experiment(name=experiment_name)

with mlflow.start_run(experiment_id=exp_id):
    # Use mlflow pyfunc log_model to log the custom model not supported by mlflow
    mlflow.sklearn.log_model(
        model,
        "external_sklearn_model",
        serialization_format="cloudpickle",
        registered_model_name="elasticnet-regressor-external",
    )

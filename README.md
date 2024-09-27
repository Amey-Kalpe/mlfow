# mlfow

## Tracking
- Mlflow can track the hyperparameters, metrics, artifacts and model files for each run using the respective log methods:
```
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)
mlflow.sklearn.log_model(lr, "mymodel")
mlflow.log_artifact("data/test_data.csv")
```

- The **autolog** method in mlflow tracks the parameters, metrics and artifacts automatically for the data science package used. The currently supported packages are: Fastai, Gluon, Keras/TensorFlow, LangChain, LlamaIndex, LightGBM, OpenAI, Paddle, PySpark, PyTorch, Scikit-learn, Spark, Statsmodels, XGBoost.
- In case of custom models, using the log methods mentioned above is recommended.
- Every run that is performed for a model can be viewed in detail on the mlflow UI:
```
terminal -> mlflow ui
```

- Start the dedicated tracking server using the below command and set the server URI in the code using:
```
terminal -> mlflow server --host 127.0.0.1 --port 5000
code -> mlflow.set_tracking_uri("http://127.0.0.1:5000")
```
Read more here: [mlflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html)


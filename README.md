# mlfow

# Tracking
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

### Autolog
- The [**autolog**](https://mlflow.org/docs/latest/tracking/autolog.html) method in mlflow tracks the parameters, metrics and artifacts automatically for the data science package used. The currently supported packages are: Fastai, Gluon, Keras/TensorFlow, LangChain, LlamaIndex, LightGBM, OpenAI, Paddle, PySpark, PyTorch, Scikit-learn, Spark, Statsmodels, XGBoost.
- In case of custom models, using the log methods mentioned above is recommended.
- Every run that is performed for a model can be viewed in detail on the mlflow UI:
```
terminal -> mlflow ui
```

- Start the dedicated tracking server using the below command and set the server URI in the code using:
``` bash
terminal -> mlflow server --host 127.0.0.1 --port 5000
```
``` python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```
Read more here: [mlflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html)

# Models
- MLFlow's Model component is a standard format which allows for reusability of models in different environment through model packages.
- It is similar to Docker in the sense that all the information of the dependencies along with the model itself is packaged.
- It provides a user friendly UI that allows to see model versions, history, sharing the model with other team members and promote models to production.
- It provides flexibility through MLFlow's *Model Serving Component* which allows deploying models for real-time inference, batch inference or edge deployment (running the model on-device).

### Model Storage Format
#### Directory Structure
- The default storage format for mlfow experiments is directory structure, i.e the mlruns folder that is created after first experiment is run using mlflow.
- The directory structure when specifying a custom folder for artifacts is as follows:
``` 
artifacts/
├── conda.yaml
├── model.pkl
├── MLmodel
├── mlflow_model.yaml
└── python_env.yml
```
- conda.yaml:
    - This file is created when you use MLflow to log your model with a conda environment.
    - It specifies the exact dependencies and their versions that were used to train and package the model.
    - This ensures reproducibility by recreating the same environment when deploying or serving the model.
- model.pkl:
    - This file contains the serialized sklearn model object.
    - It's the core component of the model and can be loaded and used to make predictions.
- MLmodel:
    - The MLmodel file is a metadata file that provides information about the model, such as its flavor (e.g., sklearn), version, and any custom tags or parameters.
    - It is typically located in the root directory of the MLflow artifact folder.
- mlflow_model.yaml:
    - This file is a metadata file that describes the model.
    - It contains information like the model's flavor (e.g., sklearn), the model's version, and any custom tags or parameters.
- python_env.yml:
    - This file is created when you use MLflow to log your model with a plain Python environment.
    - It specifies the requirements.txt file that lists the dependencies used to train and package the model.
    - This ensures reproducibility by recreating the same environment when deploying or serving the model.
- If custom artifact location not specified:
~~~
mlruns/
├── **<experiment_id>**
│   ├── **<run_id>**
│   │   ├── artifacts
│   │   ├── metrics
│   │   ├── tags
│   │   ├── meta.yaml
│   │   └── params.yaml
│   ├── **<run_id>**
│   │   ├── artifacts
│   │   ├── metrics
│   │   ├── tags
│   │   ├── meta.yaml
│   │   └── params.yaml
│   └── ...
├── **<experiment_id>**
│   ├── **<run_id>**
│   │   ├── artifacts
│   │   ├── metrics
│   │   ├── tags
│   │   ├── meta.yaml
│   │   └── params.yaml
│   ├── **<run_id>**
│   │   ├── artifacts
│   │   ├── metrics
│   │   ├── tags
│   │   ├── meta.yaml
│   │   └── params.yaml
│   └── ...
└── ...
~~~

### Model Signature
- The model signature in the **Model** component of mlflow defines the type and shape of the input and output data that is used/expected by the model.
- For example:
``` yml
signature:
  inputs: '[{"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": [-1, 4]}}]'
  outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1]}}]'
  params: null
```
- There are two types of signatures created by mlflow, column-based and tensor-based.
- The above example is of tensor-based signature, which are only supported by the deep learning flavors of mlflow, i.e. TensorFlow, PyTorch, Keras, Onnx and Gloun.
- In column-based signature, each column of the input data is treated as a separate feature and is given a unique name and data type, such as double, integer and string.

### Model Signature Enforcement
- To maintain consistency in model input and output in different environments, mlflow provides model signature enforcement which is used for validating the input and output data against the expected model signature.
- Three types of enforcement methods are provided by mlflow:
    - Signature Enforcement:
        - Checks the inputs provided to the model is of expected signature.
        - Works only with MLFlow deployment tools or when lodaing models as python_function. *Not applied to models that are loaded in their native format.*
        - Important in production environments where consistency and accuracy are paramount.
    - Name-Ordering enforcement:
        - This ensures input names provided to the model match the expected names in the signature.
        - Raise exception for missing inputs. Ignore the extra inputs.
        - If the input schema does not have input names, matching is done by position.
        - Important when models are being used in complex workflows or pipelines.
    - Input Type Enforcement:
        - This ensures **input types** provided to the model match the expected signature.
        - For column based signatures, mlflow performs safe type conversions if necessary. Valid conversion: int -> long or int -> double, but not long -> double.
        - For tensor based signatures, type checking is strict, no conversion is made.

# Model API
The MLflow Model API is a powerful tool that simplifies the deployment and management of machine learning models. It provides a consistent interface for deploying models to various serving platforms, regardless of the underlying framework or language used to train the model.

### Key features and benefits:

- Model packaging: The API allows you to package your trained models into a standardized format that can be easily deployed to different environments.
- Model serving: It provides integration with popular model serving platforms like TensorFlow Serving, PyTorch Serving, and MLflow's own model server.
- Model deployment: The API makes it simple to deploy packaged models to various cloud platforms, on-premises infrastructure, or edge devices.
- Model management: The API helps you manage the lifecycle of your models, including versioning, tracking, and serving different versions.
- Framework agnostic: The API supports a wide range of machine learning frameworks, including TensorFlow, PyTorch, Scikit-learn, and more.

## Difference Between `save_model` and `log_model` in MLflow

In `MLflow`, both `save_model` and `log_model` are used to persist models, but they differ in how and where the model is stored. Below is a comparison of the two methods:

### 1. `mlflow.save_model()`
- **Purpose**: Saves the model locally to a specific file path.
- **Where it stores**: The model is saved on the local filesystem (or a specified directory in the file system).
- **Usage scenario**: Use `save_model()` when you want to store the model for local access, without tracking it in the MLflow experiment or run metadata.
- **Example**:
    ```python
    mlflow.sklearn.save_model(model, path="models/my_model")
    ```
- **Key points**:
  - The model is not associated with an MLflow experiment or run.
  - The saved model can be loaded later using `mlflow.<flavor>.load_model()`.

### 2. `mlflow.log_model()`
- **Purpose**: Logs the model as an artifact in an MLflow run, making it trackable and accessible via the MLflow UI.
- **Where it stores**: The model is logged to the MLflow tracking server under a specific run, and it’s stored as part of that run’s artifacts (which can be stored locally, in cloud storage, or other artifact locations configured for MLflow).
- **Usage scenario**: Use `log_model()` when you want to store the model in MLflow’s tracking system so that it can be versioned, tracked, and viewed in the MLflow UI.
- **Example**:
    ```python
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path="model")
    ```

## Load Models
### 3. `mlflow.load_model()`
The load_model method in MLflow is used to load a previously saved or logged model from a specified path or model URI. It returns a trained model object that can be used for making predictions.

### Here's how it works:

- Specify the model path or URI: You provide the path to the directory where the model is saved or the model URI if it's logged to MLflow Tracking.
- Load the model: The load_model method loads the model object from the specified location.
- Return the model: The method returns the loaded model object, which can be used for inference.

- The load_model method can load models saved using either save_model or log_model.
- If you use log_model, you can provide the model's run ID or name to load a specific version.
- **Example**:
    ``` python
    import mlflow

    # Load a previously saved model
    model = mlflow.sklearn.load_model("path/to/your/model")

    # Make predictions
    predictions = model.predict(new_data)
    ```

## Model Evaluation
The [**evaluate**](https://mlflow.org/docs/latest/models.html#performing-model-validation) method in MLflow is crucial for assessing the performance of machine learning models:
- It helps data scientists and ML engineers measure, interpret, and explain how well their models are performing on new datasets.
- This step is essential for validating the accuracy, reliability, and efficiency of models before deploying them.

### Important Parameters in the evaluate Method:
- **Model to Evaluate**: This can be an MLflow pyfunc model, a URI pointing to a registered MLflow model, or any Python callable representing your model (e.g., a HuggingFace text summarization pipeline).
- **Metrics**: The metrics to compute during evaluation. MLflow supports various metrics, including those specific to large language models (LLMs).
- **Evaluation Data**: The data on which the model is evaluated. This can be a Pandas DataFrame, a Python list, a NumPy array, or an mlflow.data.dataset.Dataset instance.

#### Different runs from an experiment can be compared using the mlflow UI, to get insights into the model performance with differing configurations.



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

# Model Evaluation
The [**evaluate**](https://mlflow.org/docs/latest/models.html#performing-model-validation) method in MLflow is crucial for assessing the performance of machine learning models:
- It helps data scientists and ML engineers measure, interpret, and explain how well their models are performing on new datasets.
- This step is essential for validating the accuracy, reliability, and efficiency of models before deploying them.

### Important Parameters in the evaluate Method:
- **Model to Evaluate**: This can be an MLflow pyfunc model, a URI pointing to a registered MLflow model, or any Python callable representing your model (e.g., a HuggingFace text summarization pipeline).
- **Metrics**: The metrics to compute during evaluation. MLflow supports various metrics, including those specific to large language models (LLMs).
- **Evaluation Data**: The data on which the model is evaluated. This can be a Pandas DataFrame, a Python list, a NumPy array, or an mlflow.data.dataset.Dataset instance.

#### Different runs from an experiment can be compared using the mlflow UI, to get insights into the model performance with differing configurations.

### Baseline Models

**Using a baseline model and setting thresholds are crucial practices in machine learning for several reasons:**

- **Benchmarking Performance:**
Baseline Model: It serves as a point of reference to compare the performance of your advanced models against a simpler, often more interpretable model.
Thresholds: They provide a clear target or minimum requirement for your model's performance. This helps in assessing whether your model meets the desired standards or improvements over the baseline.
- **Evaluating Model Effectiveness:**
Baseline Model: It establishes a basic level of performance that your machine learning model should aim to surpass. This baseline is typically a simpler model or a naive approach.
Thresholds: They define what constitutes a meaningful improvement over this baseline. For instance, if your baseline accuracy is 75%, you might set a threshold that your new model should achieve at least 80% accuracy to be considered an improvement.
- **Decision-Making and Resource Allocation:**
Baseline Model: It helps in decision-making by providing a point of comparison. For example, if your advanced model does not significantly outperform the baseline, it may not justify the additional complexity and computational resources required.
Thresholds: They assist in resource allocation by setting clear goals. Teams can focus efforts on models that meet or exceed predefined thresholds, thereby optimizing time and resources.
- **Tracking Progress:**
Baseline Model: It serves as a baseline against which you can track the progress of your model iterations or improvements over time.
Thresholds: They provide a quantitative measure of progress. Meeting or exceeding thresholds indicates measurable improvement, facilitating iterative development and experimentation.
- **Risk Mitigation:**
Baseline Model: It mitigates the risk of overfitting or overcomplicating your model. By comparing against a simpler baseline, you ensure that your advanced model truly adds value and generalizes well.
Thresholds: They mitigate the risk of deploying models that do not meet minimum performance standards. Setting thresholds ensures that only models meeting acceptable performance levels proceed to deployment or further development stages.

### Practical Application
Example: In a classification task for disease prediction, a baseline logistic regression model might achieve 80% accuracy. Setting a threshold of 85% accuracy for your new model ensures that it provides a significant improvement. If your new model achieves 87% accuracy, it demonstrates a measurable enhancement over the baseline.

To evaluate a model using the MLflow API, you can use the `mlflow.evaluate()` function, which provides a flexible way to evaluate models on a given dataset. This includes generating metrics, computing model performance, and comparing with a baseline model if provided.

### 1. **Evaluating a Model Without a Baseline**
If you want to evaluate a model without comparing it to a baseline, you can use `mlflow.evaluate()` directly on the model and dataset.

#### Example: Evaluating a Model Without a Baseline

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a model (e.g., RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log the model to MLflow
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Evaluate the model
    result = mlflow.evaluate(
        model=model, 
        data=(X_test, y_test), 
        targets="y_true", 
        model_type="classifier", 
        evaluator_config={"metric_prefix": "eval_", "log_model_explainability": True}
    )

    # Output the evaluation metrics
    print(result.metrics)
```

In this example:
- **`data`**: The test dataset and corresponding target labels.
- **`model_type`**: Specifies whether it's a `classifier` or `regressor`.
- **`evaluator_config`**: Allows for custom evaluation options (e.g., whether to log explainability).

### 2. **Evaluating a Model with a Baseline**
You can compare a model's performance with a baseline model by providing the `baseline_model` argument to the `mlflow.evaluate()` function. The baseline model could be a simple model (e.g., a rule-based or default model) to help contextualize the performance of your trained model.

#### Example: Evaluating a Model with a Baseline

```python
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a model (RandomForestClassifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Train a baseline model (DummyClassifier)
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

# Log the models to MLflow
with mlflow.start_run() as run:
    # Log the trained model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log the baseline model
    mlflow.sklearn.log_model(baseline_model, artifact_path="baseline_model")

    # Evaluate the model against the baseline
    result = mlflow.evaluate(
        model=model,
        data=(X_test, y_test),
        targets="y_true",
        model_type="classifier",
        baseline_model=baseline_model,
        evaluator_config={"metric_prefix": "eval_", "log_model_explainability": True}
    )

    # Output the evaluation metrics
    print("Model metrics:", result.metrics)
    print("Baseline metrics:", result.baseline_model_metrics)
```

In this example:
- **`baseline_model`**: A `DummyClassifier` is used to provide a baseline for comparison.
- **`baseline_model_metrics`**: After evaluation, you can compare the model's performance with the baseline's performance.
  
### Key Parameters in `mlflow.evaluate()`
- **`model`**: The model to be evaluated.
- **`data`**: A tuple consisting of `(X_test, y_test)`—features and target labels for evaluation.
- **`targets`**: The ground truth labels for the data.
- **`model_type`**: Type of model being evaluated, either `classifier` or `regressor`.
- **`baseline_model`**: (Optional) A baseline model to compare the evaluated model against.
- **`evaluator_config`**: A configuration dictionary to customize the evaluation process, such as logging model explainability or adding custom metrics.

This approach allows you to log metrics for both the main model and the baseline, enabling a direct comparison.

# Model Registry

The Model Registry in MLflow is a feature designed to manage the lifecycle of machine learning models from experimentation through to production deployment. It provides a centralized hub where data scientists, machine learning engineers, and other stakeholders can collaborate, track, and manage models effectively.

## Key Features and Concepts:
- Model Versioning:
    - Version Control: MLflow Model Registry allows you to version control models. Each time a model is registered or updated, a new version is created. This enables tracking changes over time and facilitates reproducibility.
    - Snapshotting: Models are snapshot at registration time, ensuring that you can always reproduce the exact model configuration and artifacts associated with a particular version.

- Lifecycle Management:
    - Stages: Models progress through predefined stages in their lifecycle: Staging, Production, Archived, and optionally Experimental. Each stage signifies the model's status and readiness for deployment.
    - Transition: Models can be transitioned between stages based on validation, testing, and approval workflows, ensuring proper governance and control over model deployments.

- Collaboration and Governance:
    - Permissions: Access control mechanisms ensure that only authorized users can register, update, transition, or deploy models.
    - Audit Trail: Detailed logging and audit trail functionalities track who performed which actions on the models, enhancing transparency and compliance.

- Model Deployment:
    - Integration with Deployment Tools: MLflow Model Registry integrates with deployment tools such as MLflow Projects, MLflow Models, and external deployment platforms (like Kubernetes, AWS SageMaker, Azure ML) for streamlined deployment of registered models.
    - Versioned Deployment: Deploying a specific version of a model ensures consistency and reproducibility across different environments.

- Experimentation and Comparison:
    - Model Comparison: Compare metrics, parameters, and other attributes across different model versions to evaluate performance improvements or regressions.
    - Experiment Tracking Integration: Seamlessly integrates with MLflow Experiment Tracking, allowing you to register successful experiments as models directly into the registry.

## Typical Workflow:
- **Model Development:** Data scientists train and validate models using MLflow Tracking. Once a model meets certain criteria (e.g., validation performance), they register the model into the Model Registry.

- **Review and Approval**: Model reviewers (e.g., domain experts, stakeholders) examine model performance, documentation, and associated artifacts. They approve the model to move it from staging to production or provide feedback for improvements.

- **Deployment**: Approved models are deployed into production environments. MLflow Model Registry ensures that the deployed model matches the registered version, reducing the risk of model drift.

- **Monitoring and Maintenance**: Continuously monitor deployed models for performance metrics, retraining needs, and potential issues. Model Registry facilitates updating, retiring, or archiving models as necessary.

## Benefits:
- Centralized Management: Provides a single source of truth for all models, enhancing collaboration and governance.
- Versioning and Reproducibility: Tracks model versions and their artifacts, ensuring reproducibility and auditability.
- Governance and Compliance: Implements access controls, audit trails, and approval workflows for regulatory compliance and organizational standards.
- Deployment Consistency: Ensures consistency between registered models and deployed instances, reducing deployment risks and improving operational reliability.

## Example Use Cases:
- Financial Services: Manage and deploy models for risk assessment, fraud detection, and customer segmentation.
- Healthcare: Track and deploy models for patient diagnostics, disease prediction, and treatment planning.
- Retail: Deploy models for demand forecasting, personalized marketing, and customer churn prediction.

## Methods of registering a trained model
MLflow provides several ways to register a machine learning model, which allows you to manage models and their versions within the MLflow Model Registry. Here are the primary ways to register a model using the MLflow API:

### 1. **Automatic Registration During Model Logging**
You can log a model using `mlflow.<flavor>.log_model()` (such as `mlflow.sklearn.log_model()` for scikit-learn models) and automatically register it in the Model Registry.

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Train a model (e.g., using scikit-learn)
    model = ...
    
    # Log the model and register it
    mlflow.sklearn.log_model(model, "model", registered_model_name="MyModel")
```
- **`registered_model_name`**: This argument registers the model under the specified name, creating a new model or versioning an existing one.

### 2. **Registering a Model Using `mlflow.register_model()`**
If you have an already logged model (e.g., from a previous run), you can explicitly register it with the `mlflow.register_model()` API.

```python
import mlflow

# Log the model and retrieve the URI of the logged model
logged_model_uri = "runs:/<run-id>/model"

# Register the model
model_version = mlflow.register_model(model_uri=logged_model_uri, name="MyModel")
```
- **`model_uri`**: The URI path to the logged model (can be obtained from `mlflow.<flavor>.log_model()`).
- **`name`**: The name of the model in the registry.

### 3. **Manual Registration in the UI**
If you've logged a model but haven't registered it yet, you can manually register it via the MLflow UI:
   - Go to the "Models" tab in the MLflow UI.
   - Select "Register model" and choose the logged model to register it in the Model Registry.

### 4. **Registering via the `MlflowClient` API**
You can use `MlflowClient` to programmatically register models and interact with the Model Registry.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register the model
model_version = client.create_model_version(
    name="MyModel",
    source="runs:/<run-id>/model",
    run_id="<run-id>"
)
```
- **`create_model_version()`**: Registers a new version of the model with the given name, source, and run ID.

### 5. **Transitioning a Model to a Stage**
Once a model is registered, you can transition it between stages (e.g., `Staging`, `Production`).

```python
# Transition model to staging
client.transition_model_version_stage(
    name="MyModel",
    version=model_version.version,
    stage="Staging"
)
```

These methods offer flexibility in automating model registration workflows and managing model versions within the MLflow Model Registry.
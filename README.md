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

## Models
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
```
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

import mlflow


def predictor(model_uri, input_data):
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    predicted_qualities = model.predict(input_data)

    return predicted_qualities

"""
Azure Function to serve predictions from an AutoML model exported to ONNX.

https://github.com/FullStackWithLawrence/azureml-example
./titanic-survival-app/function_app.py

WHY ONNX:
- A pickled AutoML model (model.pkl) can only be loaded with Microsoft's AutoML
  runtime packages (azureml-automl-runtime, azureml-training-tabular), which
  don't support Python later than 3.11.
- AutoML also saves an ONNX copy of the model when the job runs with
  enable_onnx_compatible_models (AzureAIMLStudioAuthoringAutomatedML does this
  by default). onnxruntime runs that copy on Python 3.14 with no AutoML code.

EXECUTION ENVIRONMENT:
- This code runs on Azure Functions cloud servers (NOT on your local computer),
  Python 3.14 on the Flex Consumption plan.
- Deployed via 'func azure functionapp publish' command from your local machine
- Accessible globally via HTTPS endpoint: https://<function-app-name>.azurewebsites.net/api/predict

DEPLOYMENT FLOW (see docs/AZURE_DEPLOYMENT.md):
1. Model downloaded locally: 'az ml model download' → ./automl-model/<model-name>/model.onnx
2. Entire titanic-survival-app/ folder uploaded to Azure via 'func azure functionapp publish'
3. Azure Functions runtime loads this file and routes HTTP requests to predict()

MODEL LOADING:
- The model is loaded once, on the first request, and reused afterwards.
- It's the first automl-model/**/model.onnx file, packaged and uploaded with this code.

REQUEST AND RESPONSE:
- POST {"data": [{"Age": 22.0, "Sex": 1, ...}, ...]}: one object per passenger,
  keyed by the model's input (column) names. A list of values in the model's
  input order works too: {"data": [[22.0, 1, ...]]}.
- Returns {"prediction": [0, ...]}: one label per passenger.
"""

import glob
import json
import logging
import os
from functools import cache

import azure.functions as func
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_GLOB = os.path.join(HERE, "automl-model", "**", "model.onnx")

# ONNX input type -> the numpy type to send it. AutoML gives each column its own input.
ONNX_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
    "tensor(string)": np.object_,
}

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@cache
def load_model() -> ort.InferenceSession:
    """
    Load the ONNX model, once.

    Returns:
        ort.InferenceSession: The model, ready to run

    Raises:
        FileNotFoundError: If there is no model.onnx under automl-model/
    """
    model_paths = sorted(glob.glob(MODEL_GLOB, recursive=True))
    if not model_paths:
        raise FileNotFoundError("No model.onnx under automl-model/. See docs/AZURE_DEPLOYMENT.md, step 1.")
    logging.info("Loading model: %s", model_paths[0])
    return ort.InferenceSession(model_paths[0], providers=["CPUExecutionProvider"])


def column_values(rows: list, input_names: list[str], position: int) -> list:
    """
    Get one column's values from the request's rows.

    Args:
        rows: One dict per passenger keyed by input name, or one list per passenger in input order
        input_names: The model's input names, in order
        position: Which input to get

    Returns:
        list: The column's value from each row

    Raises:
        ValueError: If a row has no value for the column
    """
    name = input_names[position]
    values = []
    for row_number, row in enumerate(rows):
        if isinstance(row, dict):
            if name not in row:
                raise ValueError(f"Row {row_number} has no value for '{name}'")
            values.append(row[name])
        else:
            if len(row) != len(input_names):
                expected = ", ".join(input_names)
                raise ValueError(f"Each row needs {len(input_names)} values, in this order: {expected}")
            values.append(row[position])
    return values


def to_model_inputs(session: ort.InferenceSession, rows: list) -> dict[str, np.ndarray]:
    """
    Turn the request's rows into the model's inputs: one column per input, typed as the model expects.

    Args:
        session: The ONNX model
        rows: One dict per passenger keyed by input name, or one list per passenger in input order

    Returns:
        dict: input name -> column of values, shaped (number of rows, 1)

    Raises:
        ValueError: If the rows are empty, or don't match the model's inputs
    """
    model_inputs = session.get_inputs()
    input_names = [model_input.name for model_input in model_inputs]
    if not isinstance(rows, list) or not rows:
        expected = ", ".join(input_names)
        raise ValueError(f"'data' must be a non-empty list of rows. Each row needs: {expected}")

    columns = {}
    for position, model_input in enumerate(model_inputs):
        dtype = ONNX_DTYPES.get(model_input.type)
        if dtype is None:
            raise ValueError(f"Input '{model_input.name}' has a type this function doesn't handle: {model_input.type}")

        values = column_values(rows, input_names, position)
        if dtype is np.object_:
            values = [str(value) for value in values]

        column: np.ndarray = np.array(values, dtype=dtype)
        columns[model_input.name] = column.reshape(-1, 1)

    return columns


def json_response(body: dict, status_code: int = 200) -> func.HttpResponse:
    """Return body as a JSON HTTP response."""
    return func.HttpResponse(json.dumps(body), status_code=status_code, mimetype="application/json")


@app.route(route="predict", methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """Predict survival for each passenger in the request."""
    try:
        session = load_model()
    except FileNotFoundError as e:
        logging.error("Model not loaded: %s", e)
        return json_response({"error": "The model is not available"}, status_code=500)

    try:
        body = req.get_json()
        rows = body["data"]
        model_inputs = to_model_inputs(session, rows)

        # The first output is the predicted label, one per row
        outputs = session.run(None, model_inputs)
        labels = np.asarray(outputs[0]).ravel()
        return json_response({"prediction": labels.tolist()})
    except Exception as e:  # pylint: disable=W0718
        return json_response({"error": str(e)}, status_code=400)

"""
https://github.com/FullStackWithLawrence/azureml-example
./titanic-survival-app/__init__.py

Azure Function to serve predictions from a pre-trained AutoML model.

EXECUTION ENVIRONMENT:
- This code runs on Azure Functions cloud servers (NOT on your local computer)
- Deployed via 'func azure functionapp publish' command from your local machine
- Accessible globally via HTTPS endpoint: https://titanic-survival-app.azurewebsites.net

DEPLOYMENT FLOW:
1. Model downloaded locally: 'az ml model download' → ./automl-model/
    az ml model download \
        --name titanic-survival \
        --version 1 \
        --resource-group ubc-cdl10 \
        --workspace-name UBC-CDL10 \
        --download-path path/to/this/repo/titanic-survival-app/automl-model/

2. Function code created locally in ./titanic-survival-app/__init__.py (this file)

3. Entire titanic-survival-app/ folder uploaded to Azure via 'func azure functionapp publish'
    cd titanic-survival-app
    func azure functionapp publish titanic-survival-app

4. Azure Functions runtime loads and executes this code on their servers

MODEL LOADING:
- The model.pkl file is loaded ONCE when the Azure Function starts up (cold start)
- Model file exists on Azure servers because it was packaged and uploaded with this code
- Path './automl-model/model.pkl' is relative to this file's location on Azure servers

RUNTIME BEHAVIOR:
- When HTTP requests arrive at the Azure endpoint, Azure executes main() function
- Input data comes from HTTP POST requests with JSON payload
- Predictions are computed using the pre-loaded model on Azure servers
- Results are returned as JSON HTTP responses

This function expects input data in JSON format and returns predictions in JSON format.
"""

import json
import os
import pickle

import azure.functions as func
import pandas as pd


# Load model once at startup
model_path = os.path.join(os.path.dirname(__file__), "./automl-model/model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)  # nosec B301


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get input data
        req_body = req.get_json()
        input_data = pd.DataFrame(req_body["data"])

        # Make prediction
        prediction = model.predict(input_data)

        # Return result
        return func.HttpResponse(json.dumps({"prediction": prediction.tolist()}), mimetype="application/json")
    # pylint: disable=W0718
    except Exception as e:
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=400)

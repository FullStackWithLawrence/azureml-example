"""
Test the titanic-survival-app Azure Function, offline.

The tests build a tiny ONNX model that stands in for the AutoML model, so they
run the real onnxruntime code path without Azure or a downloaded model.
"""

import importlib.util
import json
import os
import tempfile
from unittest.mock import patch

import azure.functions as func
import onnx
from onnx import TensorProto, helper

from .test_base import AzureMLTestBase

HERE = os.path.abspath(os.path.dirname(__file__))
FUNCTION_APP_PATH = os.path.join(HERE, "..", "..", "titanic-survival-app", "function_app.py")


def load_function_app():
    """Import titanic-survival-app/function_app.py.

    The folder name has a hyphen, so `import` can't reach it.
    """
    spec = importlib.util.spec_from_file_location("titanic_survival_function_app", FUNCTION_APP_PATH)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_test_model(path: str):
    """
    Save a tiny ONNX model shaped like an AutoML one: one input per column.

    It predicts 1 when Age + Sex > 30, else 0, and passes Name through as a
    second output, which tests the string input type.
    """
    nodes = [
        helper.make_node("Cast", ["Sex"], ["sex_float"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["Age", "sex_float"], ["total"]),
        helper.make_node("Greater", ["total", "threshold"], ["is_over"]),
        helper.make_node("Cast", ["is_over"], ["label"], to=TensorProto.INT64),
        helper.make_node("Identity", ["Name"], ["name_out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "titanic-test-model",
        inputs=[
            helper.make_tensor_value_info("Age", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("Sex", TensorProto.INT64, [None, 1]),
            helper.make_tensor_value_info("Name", TensorProto.STRING, [None, 1]),
        ],
        outputs=[
            helper.make_tensor_value_info("label", TensorProto.INT64, [None, 1]),
            helper.make_tensor_value_info("name_out", TensorProto.STRING, [None, 1]),
        ],
        initializer=[helper.make_tensor("threshold", TensorProto.FLOAT, [], [30.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def post(function_app, body) -> func.HttpResponse:
    """POST body, as JSON unless it's already bytes, to the predict function."""
    data = body
    if not isinstance(body, bytes):
        data = json.dumps(body).encode()

    request = func.HttpRequest(method="POST", url="/api/predict", body=data)
    return function_app.predict(request)


class AzureMLTestTitanicSurvivalApp(AzureMLTestBase):
    """Tests of the titanic-survival-app Azure Function."""

    @classmethod
    def setUpClass(cls):
        """Build the test model in a temporary automl-model folder, laid out like 'az ml model download' does."""
        super().setUpClass()
        cls.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        model_dir = os.path.join(cls.temp_dir.name, "automl-model", "titanic-survival-onnx")
        os.makedirs(model_dir)
        build_test_model(os.path.join(model_dir, "model.onnx"))
        cls.function_app = load_function_app()

    @classmethod
    def tearDownClass(cls):
        """Delete the test model."""
        cls.temp_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        """Point the function at the test model, and forget any model a previous test loaded."""
        super().setUp()
        model_glob = os.path.join(self.temp_dir.name, "automl-model", "**", "model.onnx")
        self.enterContext(patch.object(self.function_app, "MODEL_GLOB", model_glob))
        self.function_app.load_model.cache_clear()
        self.addCleanup(self.function_app.load_model.cache_clear)

    def test_predict_named_columns(self):
        """Test that rows keyed by column name get one prediction each."""
        response = post(
            self.function_app,
            {"data": [{"Age": 40.0, "Sex": 0, "Name": "Allen"}, {"Age": 22.0, "Sex": 1, "Name": "Braund"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/json")
        self.assertEqual(json.loads(response.get_body()), {"prediction": [1, 0]})

    def test_predict_positional_rows(self):
        """Test that rows of values in the model's input order work too, and non-string names are converted."""
        response = post(self.function_app, {"data": [[40.0, 0, 123], [22.0, 1, "Braund"]]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.get_body()), {"prediction": [1, 0]})

    def test_missing_column(self):
        """Test that a row without one of the model's columns gets a 400 that names the column."""
        response = post(self.function_app, {"data": [{"Age": 40.0, "Name": "Allen"}]})

        self.assertEqual(response.status_code, 400)
        self.assertIn("'Sex'", json.loads(response.get_body())["error"])

    def test_wrong_row_length(self):
        """Test that a positional row with the wrong number of values gets a 400 that lists the columns."""
        response = post(self.function_app, {"data": [[40.0, 0]]})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Age, Sex, Name", json.loads(response.get_body())["error"])

    def test_bad_requests(self):
        """Test that invalid JSON, a missing 'data' key and empty data each get a JSON 400, not a traceback."""
        for body in (b"not json", {"rows": []}, {"data": []}):
            with self.subTest(body=body):
                response = post(self.function_app, body)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", json.loads(response.get_body()))

    def test_model_missing(self):
        """Test that a missing model gets a 500 with a generic message that doesn't reveal file paths."""
        empty_glob = os.path.join(self.temp_dir.name, "no-model-here", "**", "model.onnx")
        with patch.object(self.function_app, "MODEL_GLOB", empty_glob):
            self.function_app.load_model.cache_clear()
            response = post(self.function_app, {"data": [[40.0, 0, "Allen"]]})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(json.loads(response.get_body()), {"error": "The model is not available"})

    def test_function_is_registered(self):
        """Test that the app registers one function, predict, which is what the Functions runtime discovers."""
        self.assertEqual([f.get_function_name() for f in self.function_app.app.get_functions()], ["predict"])

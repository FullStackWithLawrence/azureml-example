"""
Test script to verify the Azure AutoML helper class, offline.

There is no live test here on purpose: an AutoML job runs a compute cluster
for up to experiment_timeout_hours, which costs real money.
"""

from logging import getLogger
from types import SimpleNamespace
from typing import cast

from azure.ai.ml.automl import ClassificationJob, ForecastingJob
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Job
from azure.core.exceptions import HttpResponseError

from azure_ai.ml_studio import AzureAIMLStudioAuthoringAutomatedML

from .test_base import AzureMLTestBase

logger = getLogger(__name__)


def fake_job(**fields) -> Job:
    """A stand-in for an Azure ML Job, with only the fields that a test needs."""
    return cast(Job, SimpleNamespace(**fields))


class AzureMLTestAutomatedML(AzureMLTestBase):
    """Tests of AzureAIMLStudioAuthoringAutomatedML."""

    def setUp(self):
        """Fake a workspace that holds data asset maths version 3 and cluster tiny-cluster."""
        super().setUp()
        self.ml_client = self.mock_azure()
        self.ml_client.data.get.return_value = SimpleNamespace(name="maths", version="3")
        self.ml_client.compute.get.return_value = AmlCompute(name="tiny-cluster")
        self.automl = AzureAIMLStudioAuthoringAutomatedML(dataset_name="maths", cluster_name="tiny-cluster")

    def test_run_automl_experiment(self):
        """Test that run_automl_experiment() submits a classification job on the MLTable data asset, with limits."""
        job = self.automl.run_automl_experiment(
            experiment_name="maths-experiment",
            target_column="G3",
            experiment_timeout_hours=0.5,
            iteration_timeout_minutes=10,
            max_concurrent_iterations=2,
            cv_folds=3,
        )

        automl_job = self.ml_client.jobs.create_or_update.call_args.args[0]
        self.assertIs(job, self.ml_client.jobs.create_or_update.return_value)
        self.assertIsInstance(automl_job, ClassificationJob)
        self.assertEqual(automl_job.compute, "tiny-cluster")
        self.assertEqual(automl_job.experiment_name, "maths-experiment")
        self.assertEqual(automl_job.target_column_name, "G3")
        self.assertEqual(automl_job.training_data.type, AssetTypes.MLTABLE)
        self.assertEqual(automl_job.training_data.path, "azureml:maths:3")
        self.assertEqual(automl_job.n_cross_validations, 3)
        self.assertEqual(automl_job.limits.timeout_minutes, 30)
        self.assertEqual(automl_job.limits.trial_timeout_minutes, 10)
        self.assertEqual(automl_job.limits.max_concurrent_trials, 2)
        self.assertTrue(automl_job.limits.enable_early_termination)
        self.assertTrue(automl_job.training.enable_onnx_compatible_models)

    def test_concurrent_trials_capped_at_cluster_nodes(self):
        """Test that run_automl_experiment() asks for no more concurrent trials than the cluster has nodes."""
        self.automl.cluster = AmlCompute(name="tiny-cluster", max_instances=1)

        self.automl.run_automl_experiment(
            experiment_name="maths-experiment", target_column="G3", max_concurrent_iterations=4
        )

        automl_job = self.ml_client.jobs.create_or_update.call_args.args[0]
        self.assertEqual(automl_job.limits.max_concurrent_trials, 1)

    def test_run_automl_experiment_failure(self):
        """Test that a job the workspace rejects is raised."""
        self.ml_client.jobs.create_or_update.side_effect = HttpResponseError("The AutoMLJob input is invalid")

        with self.assertRaises(HttpResponseError):
            self.automl.run_automl_experiment(experiment_name="maths-experiment", target_column="G3")

    def test_get_best_model_unsubmitted_job(self):
        """Test that get_best_model() raises ValueError for a job that was never submitted, without waiting."""
        with self.assertRaises(ValueError):
            self.automl.get_best_model(fake_job(name=None))
        self.ml_client.jobs.stream.assert_not_called()

    def test_register_best_model_failure(self):
        """Test that a model registration that fails, e.g. the trial saved no model, is raised."""
        finished_parent = fake_job(
            name="parent-job", status="Completed", tags={}, properties={"automl_best_child_run_id": "parent-job_7"}
        )
        best_child = fake_job(name="parent-job_7", properties={})
        self.ml_client.jobs.get.side_effect = [finished_parent, best_child]
        self.ml_client.models.create_or_update.side_effect = HttpResponseError("no such artifact")

        with self.assertRaises(HttpResponseError):
            self.automl.register_best_model(finished_parent, model_name="best-01")

    def test_run_forecasting_experiment(self):
        """Test that a forecasting job gets the time column."""
        self.automl.run_automl_experiment(
            experiment_name="sales-forecast",
            target_column="sales",
            task="forecasting",
            primary_metric="normalized_root_mean_squared_error",
            time_column_name="date",
        )

        automl_job = self.ml_client.jobs.create_or_update.call_args.args[0]
        self.assertIsInstance(automl_job, ForecastingJob)
        self.assertEqual(automl_job.forecasting_settings.time_column_name, "date")

    def test_unsupported_task(self):
        """Test that an unknown task raises ValueError before anything is submitted."""
        with self.assertRaises(ValueError):
            self.automl.run_automl_experiment(experiment_name="x", target_column="y", task="clustering")
        self.ml_client.jobs.create_or_update.assert_not_called()

    def test_register_best_model(self):
        """Test that register_best_model() waits for the job, then registers the best trial's MLflow model."""
        parent = fake_job(name="parent-job", status="Completed", tags={}, properties={})
        finished_parent = fake_job(
            name="parent-job", status="Completed", tags={"automl_best_child_run_id": "parent-job_7"}, properties={}
        )
        best_child = fake_job(name="parent-job_7", properties={"run_algorithm": "LightGBM"})
        self.ml_client.jobs.get.side_effect = [finished_parent, best_child]

        model = self.automl.register_best_model(parent, model_name="best-01", tags={"course": "demo"})

        self.ml_client.jobs.stream.assert_called_once_with("parent-job")
        registered = self.ml_client.models.create_or_update.call_args.args[0]
        self.assertEqual(registered.name, "best-01")
        self.assertEqual(registered.type, AssetTypes.MLFLOW_MODEL)
        self.assertEqual(registered.path, "azureml://jobs/parent-job_7/outputs/artifacts/outputs/mlflow-model/")
        self.assertEqual(registered.tags, {"course": "demo"})
        self.assertIs(model, self.ml_client.models.create_or_update.return_value)

    def test_register_best_onnx_model(self):
        """Test that register_best_model(model_format='onnx') registers the best trial's model.onnx file."""
        parent = fake_job(name="parent-job", status="Completed", tags={}, properties={})
        finished_parent = fake_job(
            name="parent-job", status="Completed", tags={"automl_best_child_run_id": "parent-job_7"}, properties={}
        )
        best_child = fake_job(name="parent-job_7", properties={})
        self.ml_client.jobs.get.side_effect = [finished_parent, best_child]

        self.automl.register_best_model(parent, model_name="titanic-survival-onnx", model_format="onnx")

        registered = self.ml_client.models.create_or_update.call_args.args[0]
        self.assertEqual(registered.type, AssetTypes.CUSTOM_MODEL)
        self.assertEqual(registered.path, "azureml://jobs/parent-job_7/outputs/artifacts/outputs/model.onnx")

    def test_register_unsupported_model_format(self):
        """Test that an unknown model format raises ValueError before waiting on the job."""
        job = fake_job(name="parent-job")
        with self.assertRaises(ValueError):
            self.automl.register_best_model(job, model_name="best-01", model_format="pickle")
        self.ml_client.jobs.stream.assert_not_called()

    def test_get_best_model_without_best_trial(self):
        """Test that a job that reports no best trial (e.g. it failed) raises ValueError."""
        failed = fake_job(name="parent-job", status="Failed", tags={}, properties={})
        self.ml_client.jobs.get.return_value = failed

        with self.assertRaises(ValueError):
            self.automl.get_best_model(failed)

    def test_show_run_details(self):
        """Test that show_run_details() returns the job's studio url."""
        job = fake_job(name="parent-job", studio_url="https://ml.azure.com/runs/parent-job")
        self.assertEqual(self.automl.show_run_details(job), "https://ml.azure.com/runs/parent-job")

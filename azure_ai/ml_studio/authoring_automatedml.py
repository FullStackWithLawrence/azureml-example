"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Azure AI ML Studio Authoring -> Automated ML: run AutoML jobs and register the best model.
"""

from logging import getLogger

from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Data, Job, Model

from .assets_dataset import AzureAIMLAssetsDataset
from .const import AUTOML_TASKS, MODEL_FORMATS
from .ml_studio_compute_cluster import AzureAIMLStudioComputeCluster
from .workspace import AzureAIMLWorkspace

logger = getLogger(__name__)


class AzureAIMLStudioAuthoringAutomatedML(AzureAIMLWorkspace):
    """
    Azure AI ML Studio Authoring -> AutomatedML class.

    Class to encapsulate Azure AutoML functionality.
    """

    def __init__(self, dataset_name: str, cluster_name: str, **kwargs):
        """
        Azure AI ML Studio Authoring -> AutomatedML class.

        Initialize the AzureAIMLStudioAuthoringAutomatedML class.
        """
        super().__init__()
        assets_dataset = AzureAIMLAssetsDataset(dataset_name=dataset_name, **kwargs)
        compute_cluster = AzureAIMLStudioComputeCluster(cluster_name=cluster_name, **kwargs)

        # get_or_create() returns what each constructor already fetched or created
        self.dataset: Data = assets_dataset.get_or_create()
        self.cluster: AmlCompute = compute_cluster.get_or_create()

    def run_automl_experiment(
        self,
        experiment_name: str,
        target_column: str,
        task: str = "classification",
        primary_metric: str = "accuracy",
        max_concurrent_iterations: int = 4,
        max_cores_per_iteration: int = -1,
        test_size: float = 0.2,
        cv_folds: int = 5,
        enable_early_stopping: bool = True,
        iteration_timeout_minutes: int = 20,
        experiment_timeout_hours: float = 1.0,
        time_column_name: str | None = None,
        enable_onnx_compatible_models: bool = True,
    ) -> Job:
        """
        Submit an AutoML job, and return without waiting for it to finish.

        Args:
            experiment_name: Name of the experiment that groups the job's runs
            target_column: Name of the target column for prediction
            task: ML task type ('classification', 'regression', 'forecasting')
            primary_metric: Primary metric to optimize
            max_concurrent_iterations: Maximum concurrent AutoML trials, capped at the cluster's node count
            max_cores_per_iteration: Cores per trial (-1 for all available)
            test_size: Proportion of data held out for testing the best model
            cv_folds: Number of cross-validation folds
            enable_early_stopping: Whether to end the job early when scores stop improving
            iteration_timeout_minutes: Timeout per trial in minutes
            experiment_timeout_hours: Total job timeout in hours
            time_column_name: Name of the time column (required for 'forecasting')
            enable_onnx_compatible_models: Whether to try only algorithms that AutoML can
                export to ONNX, and save each trial's model as outputs/model.onnx too.
                The ONNX model is what titanic-survival-app serves on Python 3.14.

        Returns:
            Job: The submitted AutoML job

        Raises:
            ValueError: If the task is not supported
        """
        if task not in AUTOML_TASKS:
            raise ValueError(f"Unsupported AutoML task: {task}. Use one of {', '.join(AUTOML_TASKS)}")

        create_automl_job = AUTOML_TASKS[task]
        training_data_path = f"azureml:{self.dataset.name}:{self.dataset.version}"
        training_data = Input(type=AssetTypes.MLTABLE, path=training_data_path)
        timeout_minutes = int(experiment_timeout_hours * 60)

        # Each trial runs on its own node, and AutoML rejects a job that asks
        # for more concurrent trials than the cluster has nodes.
        max_nodes = self.cluster.max_instances or 1
        max_concurrent_trials = min(max_concurrent_iterations, max_nodes)

        try:
            # Configure the AutoML job. Featurization is automatic by default.
            automl_job = create_automl_job(
                compute=self.cluster.name,
                experiment_name=experiment_name,
                training_data=training_data,
                target_column_name=target_column,
                primary_metric=primary_metric,
                n_cross_validations=cv_folds,
                test_data_size=test_size,
            )
            automl_job.set_limits(
                timeout_minutes=timeout_minutes,
                trial_timeout_minutes=iteration_timeout_minutes,
                max_concurrent_trials=max_concurrent_trials,
                max_cores_per_trial=max_cores_per_iteration,
                enable_early_termination=enable_early_stopping,
            )
            automl_job.set_training(enable_onnx_compatible_models=enable_onnx_compatible_models)
            if task == "forecasting":
                automl_job.set_forecast_settings(time_column_name=time_column_name)

            # Submit the job
            logger.info("Submitting AutoML job for experiment: %s", experiment_name)
            job = self.ml_client.jobs.create_or_update(automl_job)

            logger.info("AutoML job submitted successfully. Job name: %s", job.name)
            return job

        except Exception as e:
            logger.error("Failed to run AutoML experiment %s: %s", experiment_name, e)
            raise e

    def get_best_model(self, job: Job) -> Job:
        """
        Wait for an AutoML job to finish, then get its best trial.

        Args:
            job: The AutoML job returned by run_automl_experiment()

        Returns:
            Job: The child job (trial) that trained the best model

        Raises:
            ValueError: If the job was never submitted, or did not report a best trial
        """
        if not job.name:
            raise ValueError("The AutoML job has no name. Submit it with run_automl_experiment() first.")

        try:
            # Wait for job completion, printing its log as it runs
            self.ml_client.jobs.stream(job.name)
            job = self.ml_client.jobs.get(job.name)

            # AutoML records the best trial on the parent job, as a tag or a property
            tags = job.tags or {}
            properties = job.properties or {}
            best_child_name = tags.get("automl_best_child_run_id")
            if not best_child_name:
                best_child_name = properties.get("automl_best_child_run_id")
            if not best_child_name:
                raise ValueError(f"AutoML job {job.name} finished with status {job.status} and no best trial")

            best_job = self.ml_client.jobs.get(best_child_name)
            best_job_properties = best_job.properties or {}
            algorithm = best_job_properties.get("run_algorithm", "Unknown")
            logger.info("Best trial: %s", best_job.name)
            logger.info("Best model algorithm: %s", algorithm)

            return best_job

        except Exception as e:
            logger.error("Failed to get best model from job %s: %s", job.name, e)
            raise e

    def register_best_model(
        self,
        job: Job,
        model_name: str,
        description: str | None = None,
        tags: dict | None = None,
        model_format: str = "mlflow",
    ) -> Model:
        """
        Register the best model from an AutoML job.

        AutoML saves each trial's model in MLflow format, which batch
        endpoints can serve without a scoring script, and, when the job ran with
        enable_onnx_compatible_models, in ONNX format, which titanic-survival-app
        serves from an Azure Function.

        Args:
            job: The AutoML job returned by run_automl_experiment()
            model_name: Name to register the model with
            description: Description for the model
            tags: Tags to associate with the model
            model_format: 'mlflow' or 'onnx'

        Returns:
            Model: The registered model

        Raises:
            ValueError: If the model format is not supported
        """
        if model_format not in MODEL_FORMATS:
            raise ValueError(f"Unsupported model format: {model_format}. Use one of {', '.join(MODEL_FORMATS)}")
        artifact_path, model_type = MODEL_FORMATS[model_format]
        description = description or "Best model from AutoML experiment"
        tags = tags or {}

        try:
            best_job = self.get_best_model(job)
            logger.info(
                "Registering best %s model from job %s with name: %s, best trial: %s",
                model_format,
                job.name,
                model_name,
                best_job.name,
            )

            # Register the model
            model_path = f"azureml://jobs/{best_job.name}/outputs/artifacts/{artifact_path}"
            model = Model(
                name=model_name,
                path=model_path,
                type=model_type,
                description=description,
                tags=tags,
            )
            model = self.ml_client.models.create_or_update(model)

            logger.info("Successfully registered model: %s version %s", model.name, model.version)
            return model

        except Exception as e:
            logger.error("Failed to register model %s: %s", model_name, e)
            raise e

    def show_run_details(self, job: Job) -> str | None:
        """
        Get the link to an AutoML job in Azure ML studio.

        The studio shows the job's trials and metrics as it runs. (The v1 SDK's
        Jupyter widget has no v2 equivalent.)

        Args:
            job: The AutoML job to display

        Returns:
            str: The job's Azure ML studio url, or None if the job was never submitted
        """
        logger.info("AutoML job %s: %s", job.name, job.studio_url)
        return job.studio_url

from .automated_ml import AzureAutoML


def main():

    automl = AzureAutoML(dataset_name="student-performance-base", cluster_name="standard-cluster")

    automl.run_automl_experiment(
        experiment_name="test",
        target_column="age",
        task="prediction",
        primary_metric="accuracy",
        max_concurrent_iterations=4,
        max_cores_per_iteration=-1,
        test_size=0.1,
        cv_folds=5,
        enable_early_stopping=True,
        iteration_timeout_minutes=10,
        experiment_timeout_hours=1.0,
    )

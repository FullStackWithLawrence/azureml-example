#!/usr/bin/env python3
"""
Basic Azure ML example showing how to connect to workspace and create experiments.
"""

import os

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def basic_azureml_example():
    """Basic example of Azure ML functionality."""
    try:
        from azureml.core import Environment, Experiment, Workspace

        print("Azure ML Basic Example")
        print("=" * 30)

        # Method 1: Connect to workspace using config file
        # (requires config.json file with workspace details)
        print("Option 1: Connect using config file")
        print("  ws = Workspace.from_config()")
        print("  # Requires config.json with subscription_id, resource_group, workspace_name")

        # Method 2: Connect using environment variables
        print("\nOption 2: Connect using environment variables")
        print("  Set these environment variables:")
        print("  - AZUREML_SUBSCRIPTION_ID")
        print("  - AZUREML_RESOURCE_GROUP")
        print("  - AZUREML_WORKSPACE_NAME")

        subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
        resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
        workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")

        if all([subscription_id, resource_group, workspace_name]):
            print(f"  Found environment variables for workspace: {workspace_name}")
            # Uncomment to actually connect:
            # ws = Workspace(subscription_id, resource_group, workspace_name)
            # print(f"  Connected to: {ws.name}")
        else:
            print("  Environment variables not set")

        # Example: Creating an experiment
        print("\nExample: Creating an experiment")
        print("  experiment = Experiment(workspace=ws, name='my-experiment')")
        print("  run = experiment.start_logging()")
        print("  run.log('metric_name', value)")
        print("  run.complete()")

        # Example: Using environments
        print("\nExample: Creating a training environment")
        print("  env = Environment(name='my-env')")
        print("  env.python.conda_dependencies.add_pip_package('scikit-learn')")
        print("  env.register(workspace=ws)")

        return True

    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def training_example():
    """Example of training with Azure ML."""
    print("\nTraining Example")
    print("=" * 20)
    print("from azureml.train.sklearn import SKLearn")
    print("from azureml.core import ScriptRunConfig")
    print("")
    print("# Configure training script")
    print("config = ScriptRunConfig(")
    print("    source_directory='./src',")
    print("    script='train.py',")
    print("    compute_target='cpu-cluster',")
    print("    environment=env")
    print(")")
    print("")
    print("# Submit training job")
    print("run = experiment.submit(config)")
    print("run.wait_for_completion(show_output=True)")


def main():
    """Run examples."""
    basic_azureml_example()
    training_example()

    print("\n" + "=" * 50)
    print("✅ Azure ML is ready to use!")
    print("\nNext Steps:")
    print("1. Set up your Azure ML workspace in Azure Portal")
    print("2. Configure authentication (Azure CLI: az login)")
    print("3. Set environment variables or create config.json")
    print("4. Start building your ML pipelines!")


if __name__ == "__main__":
    main()

"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Azure AI ML Studio workspace: the connection that every other helper class inherits.
"""

from logging import getLogger

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.ai.ml.exceptions import ValidationException
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

logger = getLogger(__name__)


class AzureAIMLWorkspace:
    """
    Azure AI ML Studio Workspace class.

    Class to encapsulate Azure ML example functionality.

    Every helper class below inherits ml_client, the v2 SDK's entry point for
    all workspace operations (data, compute, jobs, models, endpoints).
    """

    ml_client: MLClient
    workspace: Workspace | None

    def __init__(self):
        """Initialize the AzureMLExample class."""
        load_dotenv()
        self.ml_client = self.connect_workspace()
        self.workspace = self.ml_client.workspaces.get(self.ml_client.workspace_name)

    def connect_workspace(self) -> MLClient:
        """
        Connect to Azure ML workspace.

        Reads config.json from the current directory or one of its parents, and
        authenticates with DefaultAzureCredential (for example, your `az login`).

        Returns:
            MLClient: A client for the workspace in config.json
        """
        try:
            return MLClient.from_config(credential=DefaultAzureCredential())
        except ValidationException as e:
            logger.error("Failed to connect to Azure ML workspace: %s", e)
            raise e

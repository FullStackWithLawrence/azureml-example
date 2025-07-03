"""
Lawrence McDaniel
https://lawrencemcdaniel.com

Get an existing workspace dataset from Azure AI Studio
usage:
    python3 -m azure_ai.commands.workspace
"""

from azure_ai.ml_studio import AzureAIMLWorkspace


if __name__ == "__main__":

    workspace = AzureAIMLWorkspace().workspace
    print(str(workspace))

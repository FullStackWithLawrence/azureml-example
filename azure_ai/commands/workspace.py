"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Connect to the Azure AI Studio workspace in config.json, and print it.
usage:
    python3 -m azure_ai.commands.workspace
"""

from azure_ai.ml_studio import AzureAIMLWorkspace

if __name__ == "__main__":

    workspace = AzureAIMLWorkspace().workspace
    print(str(workspace))

"""
Get an existing workspace dataset from Azure AI Studio
"""

from azure_ai.ml_studio import AzureAIMLWorkspace


if __name__ == "__main__":

    workspace = AzureAIMLWorkspace().workspace
    print(str(workspace))

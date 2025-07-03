"""
Get or create a compute cluster in Azure AI Studio.
"""

import json

from azure_ai.ml_studio import AzureAIMLStudioComputeCluster


if __name__ == "__main__":

    compute_cluster = AzureAIMLStudioComputeCluster(cluster_name="standard-cluster").cluster
    retval = compute_cluster.serialize()  # type: ignore[no-untyped-call]
    retval = json.dumps(retval, indent=4)
    print(retval)

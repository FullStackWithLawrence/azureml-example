"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Get or create a compute cluster in Azure AI Studio.
usage:
    python3 -m azure_ai.commands.compute_cluster cluster-name
"""

import argparse
import sys

from azure_ai.ml_studio import AzureAIMLStudioComputeCluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get or create a compute cluster.")
    parser.add_argument("cluster_name", type=str, help="A compute cluster name, e.g., tiny-cluster")
    args = parser.parse_args()

    compute_cluster = AzureAIMLStudioComputeCluster(cluster_name=args.cluster_name).get_or_create()
    compute_cluster.dump(sys.stdout)

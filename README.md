# Azure Machine Learning - Automated ML Example

[![AzureML](https://a11ybadges.com/badge?logo=azure)](https://azure.microsoft.com/en-us/products/machine-learning/)
[![Python](https://a11ybadges.com/badge?logo=python)](https://www.python.org/)
[![Unit Tests](https://github.com/FullStackWithLawrence/azureml-example/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/FullStackWithLawrence/azureml-example/actions/workflows/test.yml)
![Release Status](https://github.com/FullStackWithLawrence/azureml-example/actions/workflows/release.yml/badge.svg?branch=main)
![Auto Assign](https://github.com/FullStackWithLawrence/azureml-example/actions/workflows/auto-assign.yml/badge.svg)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![hack.d Lawrence McDaniel](https://img.shields.io/badge/hack.d-Lawrence%20McDaniel-orange.svg)](https://lawrencemcdaniel.com)

Demonstrate basic usage of Azure Machine Learning's [Automated ML](https://azure.microsoft.com/en-us/solutions/automated-machine-learning) service.

Note the following:

1. Model training is a computationally intensive task and it is not free. You'll need a paid [Azure Subscription](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account). Expect to spend in the neighborhood of $0.10/per hour ($USD) when training models.

2. This repo is currently based on Python3.9 even though this is several versions behind the latest stable version of Python. **DO NOT ARBITRARILY UPGRADE TO LATER VERSIONS OF PYTHON**. This repo is actively maintained. We monitor this. You have been warned.

## Usage

Works with Linux, Windows and macOS environments.

1. Verify project requirements: [Python 3.9](https://www.python.org/), [NPM](https://www.npmjs.com/) [Docker](https://www.docker.com/products/docker-desktop/), and [Docker Compose](https://docs.docker.com/compose/install/). Docker will need around 1 vCPU, 2Gib memory, and 30Gib of storage space.

2. Run `make` and add your credentials to the newly created `.env` file in the root of the repo.

3. Add your Azure `config.json` to the root of this project. See [Azure ML Configuration Guide](./docs/AZURE_ML_CONFIG.md) for detailed instructions on setting up an Azure Workspace and Subscription, and downloading your `config.json` file.

4. Initialize, build and run the application locally.

```console
git clone https://github.com/FullStackWithLawrence/azureml-example.git
make                # scaffold a .env file in the root of the repo
                    #
                    # ****************************
                    # STOP HERE!
                    # ****************************
                    # Review your .env file located in the project root folder.
                    #
make init           # Initialize Python virtual environment used for code auto-completion and linting
make test           # Verify that your Python virtual environment was built correctly and that
                    # azureml.core finds your config.json file.
                    #
make docker-build   # Build and configure all docker containers
make docker-run     # Run docker container
```

## Support

Please report bugs to the [GitHub Issues Page](https://github.com/FullStackWithLawrence/azureml_example-example/issues) for this project.

## Developers

Please see:

- the [Developer Setup Guide](./docs/CONTRIBUTING.md)
- and these [commit comment guidelines](./docs/SEMANTIC_VERSIONING.md) 😬😬😬 for managing CI rules for automated semantic releases.

You can also contact [Lawrence McDaniel](https://lawrencemcdaniel.com/contact) directly.

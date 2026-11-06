SHELL := /bin/bash
REPO_NAME := azureml-example

ifeq ($(OS),Windows_NT)
    PYTHON := python.exe
    ACTIVATE_VENV := venv/Scripts/activate
else
    PYTHON := python3.14
    ACTIVATE_VENV := source venv/bin/activate
endif
PIP := $(PYTHON) -m pip

# scaffold .env from .env.example on first use
ifeq ("$(wildcard .env)","")
    $(shell cp .env.example .env)
endif
include .env

.PHONY: all init test coverage test-ci clean lint analyze pre-commit pre-commit-init pre-commit-run release \
	check-python python-init python-lint python-clean python-requirements \
	docker-check docker-build docker-push docker-run docker-prune \
	help

# Default target executed when no arguments are given to make.
all: help

# initialize local development environment.
init:
	@echo "==============================================================================="
	@echo "Initializing local development environment. This will verify and set up your"
	@echo "Python virtual environment, install all 3rd-party package requirements,"
	@echo "and install the pre-commit hooks. This may take a few minutes..."
	@echo "==============================================================================="
	make check-python							# verify Python 3.14 is installed
	make python-init							# create/replace Python virtual environment and install dependencies
	make pre-commit-init						# install and configure pre-commit
	@echo ""
	@echo "==============================================================================="
	@echo "Initialization complete!"
	@echo ""
	@echo "Run 'source venv/bin/activate' to activate the Python virtual environment."
	@echo "    'make help' to see all available commands."
	@echo "==============================================================================="

test:
	$(ACTIVATE_VENV) && python -m unittest discover -s azure_ai/

# The offline tests under coverage.py, with a per-file report. GITHUB_ACTIONS=true
# skips every test that calls Azure, so this never creates billable resources.
# The options are in pyproject.toml's [tool.coverage] sections.
coverage:
	$(ACTIVATE_VENV) && GITHUB_ACTIONS=true coverage run -m unittest discover -s azure_ai/
	$(ACTIVATE_VENV) && coverage report

# CI (.github/workflows/test.yml) installs the requirements into the runner's
# Python, without a virtual environment, and sets GITHUB_ACTIONS=true itself.
# It runs the tests under coverage.py, and writes coverage.xml for Codecov.
test-ci:
	coverage run -m unittest discover -s azure_ai/
	coverage report
	coverage xml

clean:
	make python-clean

# ---------------------------------------------------------
# Code management
# ---------------------------------------------------------

lint:
	make python-lint

analyze:
	@echo "==============================================================================="
	@echo "Generating code analysis report using cloc ..."
	@echo "==============================================================================="
	cloc . --exclude-ext=svg,json,zip,csv,yml,yaml --vcs=git

pre-commit:
	make pre-commit-init
	make pre-commit-run

pre-commit-init:
	@echo "==============================================================================="
	@echo "Installing and configuring pre-commit ..."
	@echo "==============================================================================="
	pre-commit install
	pre-commit autoupdate

pre-commit-run:
	@echo "==============================================================================="
	@echo "Running pre-commit hooks on all files ..."
	@echo "==============================================================================="
	pre-commit run --all-files

release:
	@echo "==============================================================================="
	@echo "Forcing a new semantic release on GitHub by creating an empty commit and pushing to the repository ..."
	@echo "==============================================================================="
	git commit -m "fix: force a new release" --allow-empty && git push

# ---------------------------------------------------------
# Python
# ---------------------------------------------------------
check-python:
	@echo ""
	@echo "==============================================================================="
	@echo "Verifying that Python $(PYTHON) is installed ..."
	@echo "==============================================================================="
	@echo ""
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
	echo >&2 "This project requires $(PYTHON) but it's not installed.  Aborting."; \
	echo >&2 "python --version output:"; \
	python --version 2>&1; \
	exit 1; \
}

python-init:
	@echo "==============================================================================="
	@echo "Initializing Python virtual environment and installing dependencies. This may take a few minutes..."
	@echo "==============================================================================="
	make check-python
	make python-clean && \
	npm install && \
	$(PYTHON) -m venv venv && \
	$(ACTIVATE_VENV) && \
	$(PIP) install --upgrade pip && \
	$(PIP) install setuptools wheel pip-tools && \
	$(PIP) install -r requirements/local.txt

python-lint:
	@echo ""
	@echo "==============================================================================="
	@echo "Running Python linting using pre-commit and pylint ..."
	@echo "==============================================================================="
	@echo ""
	make check-python
	make pre-commit-run
	pylint ./azure_ai/

python-clean:
	@echo ""
	@echo "==============================================================================="
	@echo "Cleaning Python virtual environment and __pycache__ directories ..."
	@echo "==============================================================================="
	@echo ""
	rm -rf venv
	find ./azure_ai/ -name __pycache__ -type d -exec rm -rf {} +

python-requirements:
	@echo "==============================================================================="
	@echo "Compiling and updating Python dependency files using pip-compile ..."
	@echo "==============================================================================="
	pip install --upgrade setuptools wheel "pip-tools>=7.6.1"
	pip-compile --strip-extras requirements/in/base.in -o requirements/base.txt
	pip-compile --strip-extras requirements/in/local.in -o requirements/local.txt
	pip-compile --strip-extras requirements/in/prod.in -o requirements/prod.txt
	pip-compile --strip-extras requirements/in/test.in -o requirements/test.txt

# ---------------------------------------------------------
# Docker
# ---------------------------------------------------------
docker-check:
	@echo ""
	@echo "==============================================================================="
	@echo "Verifying that Docker is installed and running ..."
	@echo "==============================================================================="
	@echo ""
	@docker ps >/dev/null 2>&1 || { echo >&2 "This project requires Docker but it's not running.  Aborting."; exit 1; }

docker-build:
	@echo "==============================================================================="
	@echo "Building the Docker image ..."
	@echo "==============================================================================="
	make docker-check && \
	docker build -t ${DOCKERHUB_USERNAME}/${REPO_NAME} .

docker-push:
	@echo "==============================================================================="
	@echo "Pushing the Docker image to DockerHub ..."
	@echo "==============================================================================="
	make docker-check && \
	docker tag ${DOCKERHUB_USERNAME}/${REPO_NAME} ${DOCKERHUB_USERNAME}/${REPO_NAME}:latest && \
	echo "${DOCKERHUB_ACCESS_TOKEN}" | docker login --username=${DOCKERHUB_USERNAME} --password-stdin && \
	docker push ${DOCKERHUB_USERNAME}/${REPO_NAME}:latest

docker-run:
	make docker-check && \
	docker run -it -e OPENAI_API_KEY=${OPENAI_API_KEY} -e ENVIRONMENT=prod ${DOCKERHUB_USERNAME}/${REPO_NAME}:latest

docker-prune:
	@echo ""
	@echo "==============================================================================="
	@echo "Pruning Docker images, containers and build cache. "
	@echo "This will free up drive space by removing all unused Docker objects..."
	@echo "==============================================================================="
	make docker-check
	@if [ "`docker ps -aq`" ]; then \
	    docker stop $$(docker ps -aq); \
	fi
	@docker container prune -f
	@docker image prune -af
	@docker builder prune -af

# ---------------------------------------------------------
# Help
# ---------------------------------------------------------

help:
	@echo '===================================================================='
	@echo 'init                   - Initialize the local development environment'
	@echo 'test                   - Run Python unit tests in the virtual environment'
	@echo 'coverage               - Run the offline Python unit tests with a coverage report'
	@echo 'test-ci                - Run Python unit tests with coverage, without a virtual environment (CI)'
	@echo 'clean                  - Destroy the Python virtual environment and __pycache__ directories'
	@echo '<************************** Code Management **************************>'
	@echo 'lint                   - Run all code linters and formatters'
	@echo 'analyze                - Generate code analysis report using cloc'
	@echo 'pre-commit             - Install, configure and run pre-commit hooks'
	@echo 'pre-commit-init        - Install and configure pre-commit'
	@echo 'pre-commit-run         - Run all pre-commit hooks on all files'
	@echo 'release                - Force a new GitHub release'
	@echo '<************************** Python **************************>'
	@echo 'check-python           - Verify Python 3.14 is installed'
	@echo 'python-init            - Create a Python virtual environment and install dependencies'
	@echo 'python-lint            - Run Python linting using pre-commit and pylint'
	@echo 'python-clean           - Destroy the Python virtual environment and remove __pycache__ directories'
	@echo 'python-requirements    - Compile requirements/in/*.in into requirements/*.txt'
	@echo '<************************** Docker **************************>'
	@echo 'docker-check           - Verify Docker is installed and running'
	@echo 'docker-build           - Build the Docker image'
	@echo 'docker-push            - Push the Docker image to DockerHub'
	@echo 'docker-run             - Run the Docker image'
	@echo 'docker-prune           - Stop all containers and remove unused Docker objects'
	@echo '===================================================================='

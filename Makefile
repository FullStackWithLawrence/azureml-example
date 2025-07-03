SHELL := /bin/bash
REPO_NAME := azureml-example

ifeq ($(OS),Windows_NT)
    PYTHON = python.exe
    ACTIVATE_VENV = venv\Scripts\activate
else
    PYTHON = python3.9
    ACTIVATE_VENV = source ./venv/bin/activate
endif
PIP = $(PYTHON) -m pip

ifneq ("$(wildcard .env)","")
    include .env
else
    $(shell echo -e "ENVIRONMENT=dev\nAZUREML_CONFIG_PATH=./config.json\n\nDOCKERHUB_USERNAME=localhost\nDOCKERHUB_ACCESS_TOKEN=SET-ME-PLEASE\nPYTHONPATH=./venv:./azure_ai\n" >> .env)
endif

.PHONY: analyze pre-commit init lint clean test build release all python-init docker-build docker-push docker-run docker-prune help

# Default target executed when no arguments are given to make.
all: help

# -------------------------------------------------------------------------
# Install and run pre-commit hooks
# -------------------------------------------------------------------------
pre-commit:
	pre-commit install
	pre-commit autoupdate
	pre-commit run --all-files

# ---------------------------------------------------------
# create a local python virtual environment. Includes linters
# and other tools for local development.
# ---------------------------------------------------------
python-init:
	make clean
	$(PYTHON) -m venv venv && \
	$(ACTIVATE_VENV) && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements/local.txt && \
	deactivate

######################
# CORE COMMANDS
######################
init:
	make clean && \
	make python-init && \
	npm install && \
	$(ACTIVATE_VENV) && \
	$(PIP) install -r requirements/local.txt && \
	pre-commit install && \
	deactivate

test:
	$(ACTIVATE_VENV) && python -m unittest discover -s azure_ai/; \

test-ci:
	python -m unittest discover -s azure_ai/;

lint:
	isort . && \
	pre-commit run --all-files && \
	black . && \
	flake8 ./azure_ai/ && \
	pylint ./azure_ai/**/*.py

clean:
	rm -rf venv node_modules azure_ai/__pycache__ package-lock.json

analyze:
	cloc . --exclude-ext=svg,json,zip --vcs=git

release:
	git commit -m "fix: force a new release" --allow-empty && git push

######################
# DOCKER
######################
docker-build:
	docker build -t ${DOCKERHUB_USERNAME}/${REPO_NAME} .

docker-push:
	source .env && \
	docker tag ${DOCKERHUB_USERNAME}/${REPO_NAME} ${DOCKERHUB_USERNAME}/${REPO_NAME}:latest && \
	echo "${DOCKERHUB_ACCESS_TOKEN}" | docker login --username=${DOCKERHUB_USERNAME} --password-stdin && \
	docker push ${DOCKERHUB_USERNAME}/${REPO_NAME}:latest

docker-run:
	source .env && \
	docker run -it -e OPENAI_API_KEY=${OPENAI_API_KEY} -e ENVIRONMENT=prod ${DOCKERHUB_USERNAME}/${REPO_NAME}:latest


docker-prune:
	@if [ "`docker ps -aq`" ]; then \
	    docker stop $(docker ps -aq); \
	fi
	@docker container prune -f
	@docker image prune -af
	@docker builder prune -af

######################
# HELP
######################

help:
	@echo '===================================================================='
	@echo 'analyze         - generate code analysis report'
	@echo 'release         - force a new GitHub release'
	@echo 'init            - create a Python virtual environment and install prod dependencies'
	@echo 'python-init     - create a Python virtual environment with local dependencies'
	@echo 'pre-commit      - install and run pre-commit hooks'
	@echo 'test            - run Python unit tests'
	@echo 'lint            - run Python linting'
	@echo 'clean           - destroy the Python virtual environment'
	@echo 'docker-build    - build the Docker image'
	@echo 'docker-push     - push the Docker image to DockerHub'
	@echo 'docker-run      - run the Docker image'
	@echo 'docker-prune    - clean up Docker containers and images'
	@echo '===================================================================='

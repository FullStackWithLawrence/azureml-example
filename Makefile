SHELL := /bin/bash
REPO_NAME := openai-hello-world

ifeq ($(OS),Windows_NT)
    PYTHON = python.exe
    ACTIVATE_VENV = venv\Scripts\activate
else
    PYTHON = python3.9
    ACTIVATE_VENV = source venv/bin/activate
endif
PIP = $(PYTHON) -m pip

ifneq ("$(wildcard .env)","")
    include .env
else
    $(shell echo -e "ENVIRONMENT=dev\nAZUREML_CONFIG_PATH=./config.json\n\nDOCKERHUB_USERNAME=localhost\nDOCKERHUB_ACCESS_TOKEN=SET-ME-PLEASE\nPYTHONPATH=./venv:./azureml_example\n" >> .env)
endif

.PHONY: analyze pre-commit init lint clean test build release

# Default target executed when no arguments are given to make.
all: help

analyze:
	cloc . --exclude-ext=svg,json,zip --vcs=git

release:
	git commit -m "fix: force a new release" --allow-empty && git push

# -------------------------------------------------------------------------
# Install and run pre-commit hooks
# -------------------------------------------------------------------------
pre-commit:
	pre-commit install
	pre-commit autoupdate
	pre-commit run --all-files

# ---------------------------------------------------------
# create python virtual environments for prod
# ---------------------------------------------------------
python-init:
	make clean
	$(PYTHON) -m venv venv && \
	$(ACTIVATE_VENV) && \
	$(PIP) install --upgrade pip && \
	$(PIP) install -r requirements/local.txt && \
	deactivate

# ---------------------------------------------------------
# create python virtual environments for dev
# ---------------------------------------------------------
init-dev:
	make python-init && \
	npm install && \
	$(ACTIVATE_VENV) && \
	$(PIP) install -r requirements/local.txt && \
	deactivate && \
	pre-commit install

test:
	$(ACTIVATE_VENV) && which python3 && python3 --version && pip show azureml-core && python -m unittest discover -s azureml_example/

lint:
	isort . && \
	pre-commit run --all-files && \
	black . && \
	flake8 ./azureml_example/ && \
	pylint ./azureml_example/**/*.py

clean:
	rm -rf venv node_modules azureml_example/__pycache__ package-lock.json

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

docker-test:
	make docker-check && \
	docker exec smarter-app bash -c "./manage.py test"

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
	@echo 'init-dev        - install dev dependencies'
	@echo 'test            - run Python unit tests'
	@echo 'lint            - run Python linting'
	@echo 'clean           - destroy the Python virtual environment'
	@echo 'docker-build    - build the Docker image'
	@echo 'docker-push     - push the Docker image to DockerHub'
	@echo 'docker-run      - run the Docker image'

# Developer Setup Guide

[![NPM](https://a11ybadges.com/badge?logo=npm)](https://www.npmjs.com/)
[![Yarn](https://a11ybadges.com/badge?logo=yarn)](https://yarnpkg.com/)
[![Gulp](https://a11ybadges.com/badge?logo=gulp)](https://gulpjs.com/)
[![Python](https://a11ybadges.com/badge?logo=python)](https://www.python.org/)

## Quick Start

Smarter follows opinionated code style policies for most of the technologies in this repo. With that in mind, following is how to correctly setup your local development environment. Before attempting to setup this project you should ensure that the following prerequisites are installed in your local environment:

- Docker CE 25x or later
- Docker Compose
- npm: 10.4 or later
- python: 3.11 or later
- git: 2.x or later
- make: 3.8 or later

```console
git clone https://github.com/lpm0073/azureml-example.git
make         # scaffold a .env file in the root of the repo
             #
             # ****************************
             # STOP HERE!
             # ****************************
             # Add your credentials to .env located in the project
             # root folder.

make init    # initialize dev environment, build & init docker.
make build   # builds and configures all docker containers
make run     # runs all docker containers and starts a local web server on port 8000
```

To preserve your own sanity, don't spend time formatting your Python, Terraform, JS or any other source code because pre-commit invokes automatic code formatting utilities such as black, flake8 and prettier, on all local commits, and these will reformat the code in your commit based on policy configuration files found in the root of this repo.

## Good Coding Best Practices

This project demonstrates a wide variety of good coding best practices for managing mission-critical cloud-based micro services in a team environment, namely its adherence to [12-Factor Methodology](./docs/12-FACTOR.md). Please see this [Code Management Best Practices](./docs/GOOD_CODING_PRACTICE.md) for additional details.

We want to make this project more accessible to students and learners as an instructional tool while not adding undue code review workloads to anyone with merge authority for the project. To this end we've also added several pre-commit code linting and code style enforcement tools, as well as automated procedures for version maintenance of package dependencies, pull request evaluations, and semantic releases.

## Repository Setup

### .env setup

Smarter uses a **LOT** of configuration data. You'll find a pre-formatted quick-start sample .env [here](./example-dot-env) to help you get started, noting however that simply running `make` from the root of this repo will scaffold this exact file for you.

### pre-commit setup

This project uses pre-commit as a first-pass automated code review / QC process. pre-commit runs a multitude of utilities and checks for code formatting, linting, syntax checking, and ensuring that you don't accidentally push something to GitHub which you'd later regret. Broadly speaking, these checks are aimed at minimizing the extent of commits that contain various kinds of defects and stylistic imperfections that don't belong on the main branch of the project.

Note that many of the pre-commit commands are actually executed by Python which in turn is calling pip-installed packages listed in requirements/local.txt located in the root of the repo. It therefore is important that you first create the Python virtual environment using `make pre-commit`. It also is a good idea to do a complete 'dry run' of pre-commit, to ensure that your developer environment is correctly setup:

```console
make pre-commit
```

Output should look similar to the following:

![pre-commit output](./docs/img/pre-commit.png)

### GitHub Actions

This project depends heavily on GitHub Actions to automate routine activities, so that hopefully, the source code is always well-documented and easy to read, and everything works perfectly. We automate the following in this project:

- Code style and linting checks, during both pre-commit as well as triggered on pushes to the main branch
- Unit tests for Python, React and Terraform
- Docker builds
- Environment-specific deployments to Kubernetes
- Semantic Version releases
- version bumps from npm, PyPi and Terraform Registry

A typical pull request will look like the following:

![Automated pull request](./docs/img/automated-pr.png)

## Docker Setup

You can leverage Docker Community Edition and Docker Compose to stand up the entire Smarter platforn in your local development environment. This closely approximates the Kubernetes production environment in which Smarter actually runs. Everything is substantially created with two files located in the root of this repo:

- [Dockerfile](../Dockerfile): This defines the contents and configuration of the Docker container used to deploy the Smarter application, worker, and Celery Beat service in all Kubernetes environments as well as in your local Docker CE environment. Thus, think twice before pushing modifications to this file, as there could be unintended consequences.

### Unit Tests

We use `unittest` in this project. You should create relevant unit tests for your new features, sufficient to achieve a [Coverage](https://coverage.readthedocs.io/) analysis of at least 75%.

### Coverage

Coverage.py is a tool for measuring code coverage of Python programs. It monitors your program, noting which parts of the code have been executed, then analyzes the source to identify code that could have been executed but was not.

Coverage measurement is typically used to gauge the effectiveness of tests. It can show which parts of your code are being exercised by tests, and which are not.

Note the following shortcut for running a Coverage report: `make coverage`.

**Our goal for this project is to maintain an overall Coverage score of at least 80%.**

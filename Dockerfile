# Use the official Python 3.14 image from the Docker Hub.
# This runs on Debian Linux.
FROM python:3.14 AS base

############################## systempackages #################################
FROM base AS systempackages

COPY requirements/base.txt base.txt
COPY requirements/prod.txt requirements.txt

# Install any needed packages specified in requirements.txt
# Install Azure CLI and any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y curl gnupg lsb-release && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

############################## python virtual environment #################################
FROM systempackages AS venv

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


############################## app #################################
FROM venv AS app

# Set the working directory /dist
WORKDIR /dist

# Copy the current directory contents into the container at /app
COPY azure_ai /dist/azure_ai

# Set environment variables
ENV ENVIRONMENT=dev
ENV PYTHONPATH=/dist



################################# final #######################################
FROM app AS final
# Run the application when the container launches
CMD ["python", "azure_ai/commands/workspace.py"]

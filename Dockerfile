# Use the official Python 3.9 image from the Docker Hub.
# This runs on Debian Linux.
FROM --platform=linux/amd64 python:3.9 AS base

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

# dotnet runtime installation - using official Microsoft install script
RUN curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 6.0 --runtime dotnet --install-dir /usr/share/dotnet && \
    ln -s /usr/share/dotnet/dotnet /usr/local/bin/dotnet

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

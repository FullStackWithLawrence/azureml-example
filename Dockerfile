# Use the official Python 3.9 image from the Docker Hub.
# This runs on Debian Linux.
FROM python:3.9-slim

# Set the working directory /dist
WORKDIR /dist

# Copy the current directory contents into the container at /app
COPY azure_ai /dist/azure_ai
COPY requirements/base.txt base.txt
COPY requirements/prod.txt requirements.txt

# Set environment variables
ENV ENVIRONMENT=dev
ENV PYTHONPATH=/dist

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Run the application when the container launches
CMD ["python", "azure_ai/automated_ml.py"]

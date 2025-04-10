# Use an official Python runtime
FROM --platform=linux/amd64 python:3.13-slim AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
COPY ./gcp-key.json .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script and the service account key file (make sure to handle this securely)
COPY ./inference/inference.py .

# Expose the port the app runs on
EXPOSE 8080

# Run the inference server
CMD ["python", "inference.py"]

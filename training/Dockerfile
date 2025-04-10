# Use an official Python runtime
FROM --platform=linux/amd64 python:3.13-slim AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
COPY ./gcp-key.json .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training script and any additional files you need
COPY training/train.py .

# Define the command to run your training script
CMD ["python", "train.py"]

# Use an official Python runtime - lighter weight (for inference)
FROM --platform=linux/amd64 python:3.13-slim AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY ./inference/inference.py .

# Expose the port the app runs on
EXPOSE 8080

# Run the inference server
CMD ["python", "inference.py"]

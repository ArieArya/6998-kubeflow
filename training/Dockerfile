# Use an official Pytorch runtime
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS build

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script
COPY training/train.py .

# Command for training
CMD ["python", "train.py"]

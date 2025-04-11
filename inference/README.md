# Inference Workflow for Image Captioning

This README explains how to build, deploy, and run the inference service for the image captioning model on GKE. It runs a Flask-based web UI where users can upload images and receive real-time captions.

## Prerequisites

-   Docker
-   gcloud CLI (authenticated and configured)
-   kubectl CLI

## 1. Build and Push the Docker Image

From the root of this repo:

```bash
docker build -f inference/Dockerfile --platform linux/amd64 -t {your-username}/cnn-captioning-image-inference:v1 .
docker push {your-username}/cnn-captioning-image-inference:v1
```

## 2. Set Up a CPU-Only GKE Cluster

```bash
gcloud container clusters create cnn-captioning-inference \
  --zone {zone} \
  --num-nodes 1 \
  --machine-type=n1-standard-2 \
  --disk-size=100GB
```

Update your `kubectl` context:

```bash
gcloud container clusters get-credentials cnn-captioning-inference --zone {zone} --project {project-id}
```

## 3. Deploy the Inference Service

```bash
kubectl apply -f ./inference/inference.yaml
```

## 4. Monitor the Pod

```bash
kubectl get pods
kubectl logs <pod-name>
```

Replace `<pod-name>` with the actual pod name.

## 5. Get the external IP address

```bash
kubectl get svc cnn-captioning-inference-service
```

## Features

-   Web UI to upload an image.
-   Caption is displayed along with the image.
-   Served via `LoadBalancer` so it's publicly accessible.

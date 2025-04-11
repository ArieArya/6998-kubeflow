# Training Workflow for Image Captioning

This README explains how to build, deploy, and run the training workflow for the image captioning model on Google Kubernetes Engine (GKE). The model uses a ResNet encoder and an LSTM decoder (adapted from COMS 4705 NLP by Prof. Daniel Bauer) to generate captions from images. Training artifacts (model checkpoints and tokenization JSON files) are saved persistently to Google Cloud Storage (GCS).

## Prerequisites

-   Docker
-   gcloud CLI (authenticated and configured)
-   kubectl CLI

## 1. Build and Push the Docker Image

From the root of this repo:

```bash
docker build -f training/Dockerfile --platform linux/amd64 -t {your-username}/cnn-captioning-image-training:v1 .
docker push {your-username}/cnn-captioning-image-training:v1
```

(Optional: test it locally)

```bash
docker run -it --rm --entrypoint /bin/sh {your-username}/cnn-captioning-image-training:v1
```

## 2. Set Up a GPU-Enabled GKE Cluster

```bash
gcloud container clusters create cnn-captioning-training \
  --zone {zone} \
  --num-nodes 1 \
  --machine-type=n1-standard-1 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --disk-size=100GB \
  --workload-pool={project-id}.svc.id.goog \
  --workload-metadata=GKE_METADATA
```

Update your `kubectl` context:

```bash
gcloud container clusters get-credentials cnn-captioning-training --zone {zone} --project {project-id}
```

## 3. Install the Kubeflow Training Operator

```bash
kubectl apply --server-side -k "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.8.1"
kubectl get pods -n kubeflow
kubectl get crd
```

## 4. Deploy the Training Job

```bash
kubectl apply -f ./training/training.yaml
```

## 5. Monitor the Job

```bash
kubectl get pods -n kubeflow
kubectl logs -n kubeflow <pod-name>
```

Replace `<pod-name>` with the actual name of your training pod.

## Notes

-   Uses PyTorchJob from Kubeflow to manage training.
-   Requests GPU via `nvidia.com/gpu: 1`.
-   Uploads trained model and tokenizers to GCS using Workload Identity (`gcs-access` KSA).

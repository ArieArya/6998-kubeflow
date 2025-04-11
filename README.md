# Image Captioning on GKE

This repository demonstrates an end-to-end machine learning workflow for training and serving an image captioning model on Google Kubernetes Engine (GKE). The project leverages deep learning (using a ResNet encoder and an LSTM decoder) to generate captions from images. The training component runs on a GPU-enabled cluster, while the inference service (a Flask application) is deployed on a CPU-only cluster.

### Components

-   **Training:**

    -   Uses an NVIDIA GPU-enabled GKE cluster.
    -   Containerized using a CUDA-enabled PyTorch image.
    -   Training artifacts (model checkpoint and tokenization JSON files) are saved to Google Cloud Storage (GCS).
    -   Managed via a Kubeflow `PyTorchJob` custom resource.

-   **Inference:**
    -   Runs on a separate, CPU-only GKE cluster.
    -   Provides a web interface built with Flask for users to upload images and receive generated captions in real time.
    -   Deployed via a standard Kubernetes Deployment and exposed through a LoadBalancer Service.

For full details on building, deploying, and running both workflows, please refer to the following:

-   [Inference README](./inference/README.md)
-   [Training README](./training/README.md)

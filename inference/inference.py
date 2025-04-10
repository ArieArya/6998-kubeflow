import os
import json
import io
import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from torchvision import transforms, models
from google.cloud import storage
from google.oauth2 import service_account

app = Flask(__name__)

# Configuration
BUCKET_NAME = 'coms_6998_applied_ml'
MODEL_BLOB = 'models/image-captioning-model.pt'
ID_TO_WORD_BLOB = 'models/id_to_word.json'
WORD_TO_ID_BLOB = 'models/word_to_id.json'
LOCAL_MODEL_PATH = '/tmp/image-captioning-model.pt'
LOCAL_ID_TO_WORD = '/tmp/id_to_word.json'
LOCAL_WORD_TO_ID = '/tmp/word_to_id.json'

# Download helper from GCS
def download_from_gcs(bucket_name, blob_name, local_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded {blob_name} to {local_path}")

# Download model and mapping files (if not already downloaded)
download_from_gcs(BUCKET_NAME, MODEL_BLOB, LOCAL_MODEL_PATH)
download_from_gcs(BUCKET_NAME, ID_TO_WORD_BLOB, LOCAL_ID_TO_WORD)
download_from_gcs(BUCKET_NAME, WORD_TO_ID_BLOB, LOCAL_WORD_TO_ID)

# Load mapping files
with open(LOCAL_ID_TO_WORD, 'r') as f:
    id_to_word = json.load(f)
with open(LOCAL_WORD_TO_ID, 'r') as f:
    word_to_id = json.load(f)

# Note: JSON Tokenization keys
id_to_word = {int(k): v for k, v in id_to_word.items()}
word_to_id = {k: int(v) for k, v in word_to_id.items()}

# For inference, we assume the same max caption length as used in training
MAX_LEN = 40
vocab_size = len(word_to_id) + 1  # same as training

# Define the Caption Generator Model (should match training code)
class CaptionGeneratorModel(torch.nn.Module):
    def __init__(self):
        super(CaptionGeneratorModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 512)
        self.lstm = torch.nn.LSTM(1024, 512, num_layers=1, batch_first=True)
        self.output = torch.nn.Linear(512, vocab_size)

    def forward(self, img, input_seq):
        # img: (batch_size, 512)
        # Expand image encoding to (batch_size, MAX_LEN, 512)
        img_encoding = img.unsqueeze(1).expand(-1, MAX_LEN, -1)
        seq_embedding = self.embedding(input_seq)
        x = torch.cat((img_encoding, seq_embedding), dim=2)
        lstm_out, _ = self.lstm(x)
        out = self.output(lstm_out)
        return out

# Load the trained model from disk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(LOCAL_MODEL_PATH, map_location=device, weights_only=False)
model.eval()

# For encoding images, use a pretrained ResNet18 (without the final FC layer)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
modules = list(resnet.children())[:-1]  # Remove final classification layer
img_encoder = torch.nn.Sequential(*modules)
img_encoder.eval()
img_encoder.to(device)

# Preprocessing for input images (should match training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def encode_image(image: Image.Image) -> torch.Tensor:
    """Preprocess and encode an image using the ResNet encoder."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        features = img_encoder(image_tensor).squeeze()  # (512)
    return features

def generate_caption(image_tensor: torch.Tensor) -> str:
    """
    Generate a caption for an image using a simple greedy decoding approach.
    """
    # Get special token ids
    start_token = word_to_id.get("<START>", 1)
    eos_token = word_to_id.get("<EOS>", 2)

    # Initialize an input sequence with PAD tokens and set the first token as <START>
    input_seq = torch.zeros((1, MAX_LEN), dtype=torch.long).to(device)
    input_seq[0, 0] = start_token

    # Greedy decoding loop
    for t in range(1, MAX_LEN):
        with torch.no_grad():
            # model expects an image feature (of shape (batch_size, 512))
            # and the current input sequence (shape (batch_size, MAX_LEN))
            outputs = model(image_tensor.unsqueeze(0), input_seq)
            # Get logits for the current time step (t-1)
            logits = outputs[0, t-1, :]
            predicted = torch.argmax(logits).item()
        input_seq[0, t] = predicted
        if predicted == eos_token:
            break

    # Convert token ids to words, skipping special tokens as desired
    caption_tokens = []
    for token_id in input_seq[0].cpu().numpy():
        if token_id == start_token or token_id == word_to_id.get("<PAD>", 0):
            continue
        if token_id == eos_token:
            break
        caption_tokens.append(id_to_word.get(token_id, ""))
    return " ".join(caption_tokens)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to perform inference.
    Expects a multipart form-data POST with an image file under key 'image'.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file", "details": str(e)}), 400

    # Encode the image and generate a caption
    image_tensor = encode_image(image)
    caption = generate_caption(image_tensor)
    return jsonify({"caption": caption})

@app.route('/')
def home():
    # HTML with a simple UI: a file input, image preview, and caption display
    html_page = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Captioning Inference</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .preview-image { max-width: 300px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Image Captioning Inference</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageInput" name="image" accept="image/*" onchange="previewImage(event)" />
            <br>
            <img id="imagePreview" class="preview-image" style="display:none;" />
            <br>
            <button type="button" onclick="uploadImage()">Get Caption</button>
        </form>
        <h2>Caption:</h2>
        <p id="captionResult"></p>
        <script>
            function previewImage(event) {
                var imageFile = event.target.files[0];
                if (imageFile) {
                    var reader = new FileReader();
                    reader.onload = function(e) {
                        var preview = document.getElementById("imagePreview");
                        preview.src = e.target.result;
                        preview.style.display = "block";
                    }
                    reader.readAsDataURL(imageFile);
                }
            }
            function uploadImage() {
                var formData = new FormData();
                var imageFile = document.getElementById('imageInput').files[0];
                if (!imageFile) {
                    alert("Please select an image file.");
                    return;
                }
                formData.append("image", imageFile);
                fetch("/predict", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.caption) {
                        document.getElementById("captionResult").innerText = "Caption: " + data.caption;
                    } else if (data.error) {
                        document.getElementById("captionResult").innerText = "Error: " + data.error;
                    }
                })
                .catch(err => {
                    document.getElementById("captionResult").innerText = "Error: " + err;
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_page)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
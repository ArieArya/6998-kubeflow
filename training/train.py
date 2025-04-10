"""
 This training script adapted from course COMS 4705 Natural Language
 Processing (NLP) by Professor Daniel Bauer
"""

import os
import re
import json
import PIL
import torch
import torchvision
import urllib.request
import zipfile
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from google.cloud import storage
from google.oauth2 import service_account

# Define device for training
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    print("You won't be able to train the RNN decoder on a CPU, unfortunately.")
print(f"DEVICE: {DEVICE}")


#################################
### 1. Download training data ###
#################################
url = "https://storage.googleapis.com/4705_sp25_hw3/hw3data.zip"
zip_filename = "hw3data.zip"
urllib.request.urlretrieve(url, zip_filename)
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('.')
os.remove(zip_filename)


##############################
### 2. Encoding Image Data ###
##############################
def load_image_list(filename):
    with open(filename,'r') as image_list_f:
        return [line.strip() for line in image_list_f]

script_dir = os.path.dirname(os.path.abspath(__file__))
MY_DATA_DIR = "hw3data"
FLICKR_PATH = "hw3data/"
train_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.trainImages.txt'))
dev_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.devImages.txt'))
test_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.testImages.txt'))

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use the ResNet-18 Image Encoder
img_encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
lastremoved = list(img_encoder.children())[:-1]
img_encoder = torch.nn.Sequential(*lastremoved).to(DEVICE)

IMG_PATH = os.path.join(FLICKR_PATH, "Flickr8k_Dataset")
def get_image(img_name):
    image = PIL.Image.open(os.path.join(IMG_PATH, img_name))
    return preprocess(image)

def encode_images(image_list, batch_size=256):
    result = []
    for i in range(0, len(image_list), batch_size):
        print(f"images processed: {i+1}/{len(image_list)}")
        images = [get_image(img) for img in image_list[i:i+batch_size]]
        images = torch.stack(images, dim=0).to(DEVICE)
        with torch.no_grad():
            embeddings = img_encoder(images).squeeze(dim=(-1, -2))
            result.append(embeddings.cpu())
    result = torch.cat(result, dim=0)  # unroll batch embeddings
    return result

# Encode the training images
enc_images_train = encode_images(train_list)


#######################################
### 3. Preparing Image caption data ###
#######################################
def read_image_descriptions(filename):
    image_descriptions = {}

    # line can be split into three groups (img.jpg)(#number)(caption)
    re_pattern = r'(.+\.jpg(?:\.\d+)?)#(\d+)\s+(.*)'

    with open(filename,'r') as in_file:
        for line in in_file:
            match = re.match(re_pattern, line)
            if match:
                img_filename = match.group(1)
                number = int(match.group(2))
                caption = match.group(3)

                # Modify caption with start and end tokens
                caption = ['<START>'] + caption.lower().split(" ") + ['<EOS>']

                # This assumes image numbers will start from 0
                if number == 0:
                    image_descriptions[img_filename] = [0 for _ in range(5)]
                image_descriptions[img_filename][number] = caption
            else:
                print(f"Error matching line with expected format. Line: {line}")
    return image_descriptions

descriptions = read_image_descriptions(os.path.join(FLICKR_PATH, "Flickr8k.token.txt"))

# Create word indices
id_to_word = {}
id_to_word[0] = "<PAD>"
id_to_word[1] = "<START>"
id_to_word[2] = "<EOS>"
word_to_id = {}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<EOS>"] = 2

# Create a set of tokens in training data
tokens = set()
for img_filename in descriptions:
    for caption in descriptions[img_filename]:
        for token in caption:
            tokens.add(token)

# Convert to list and sort
tokens = sorted(list(tokens))

# Add to map with offset 3 (due to special tokens)
for i, token in enumerate(tokens):
    id_to_word[i+3] = token
    word_to_id[token] = i+3

# Create a Caption and Image Dataset
MAX_LEN = 40

class CaptionAndImage(Dataset):
    def __init__(self, img_list):
        self.img_data = enc_images_train
        self.img_name_to_id = dict([(i,j) for (j,i) in enumerate(img_list)])
        self.data = []

        for img_filename in img_list:
            for caption in descriptions[img_filename]:
                caption_id = [word_to_id[token] for token in caption]

                # Initialize to all pads
                input_id = [word_to_id["<PAD>"] for _ in range(MAX_LEN)]
                output_id = [word_to_id["<PAD>"] for _ in range(MAX_LEN)]

                for i in range(len(caption)-1):
                    input_id[i] = caption_id[i]
                    output_id[i] = caption_id[i+1]  # the output is right-shifted by 1

                    # Append tensors to training data
                    self.data.append((self.img_data[self.img_name_to_id[img_filename]],  # image encoding
                                      torch.tensor(input_id),
                                      torch.tensor(output_id)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,k):
        return self.data[k]

# Obtain the combined caption and image data
joint_data = CaptionAndImage(train_list)


####################################
### 4. Define the model to train ###
####################################
vocab_size = len(word_to_id)+1
class CaptionGeneratorModel(nn.Module):
    def __init__(self):
        super(CaptionGeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.lstm = nn.LSTM(1024, 512, num_layers = 1, bidirectional=False, batch_first=True)
        self.output = nn.Linear(512,vocab_size)

    def forward(self, img, input_seq):
        img_encoding = img.unsqueeze(1).expand(-1, MAX_LEN, -1)  # (batch_size, MAX_LEN, 512)
        seq_embedding = self.embedding(input_seq)  # (batch_size, MAX_LEN, 512)
        inp = torch.cat((img_encoding, seq_embedding), dim=2)  # (batch_size, MAX_LEN, 1024)
        hidden = self.lstm(inp)  # (batch_size, MAX_LEN, 512)
        out = self.output(hidden[0])  # (batch_size, MAX_LEN, vocab_size)
        return out


##########################
### 5. Train the model ###
##########################
# Bring model to DEVICE
model = CaptionGeneratorModel().to(DEVICE)
loss_function = CrossEntropyLoss(ignore_index = 0, reduction='mean')
LEARNING_RATE = 1e-03
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
loader = DataLoader(joint_data, batch_size = 128, shuffle = True)  # batch size set to 128

def train():
    """
    Train the model for one epoch.
    """
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_correct, total_predictions = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()

    for idx, batch in enumerate(loader):
        img, inputs, targets = batch
        img = img.to(DEVICE)
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Run the forward pass of the model
        logits = model(img, inputs)
        loss = loss_function(logits.transpose(2,1), targets)
        tr_loss += loss.item()
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=2)  # Predicted token labels
        not_pads = targets != 0  # Mask for non-PAD tokens
        correct = torch.sum((predictions == targets) & not_pads)
        total_correct += correct.item()
        total_predictions += not_pads.sum().item()

        if idx % 100==0:
            #torch.cuda.empty_cache() # can help if you run into memory issues
            curr_avg_loss = tr_loss/nb_tr_steps
            print(f"Current average loss: {curr_avg_loss}")

        # Run the backward pass to update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accuracy = total_correct / total_predictions if total_predictions != 0 else 0  # Avoid division by zero
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Average accuracy epoch: {epoch_accuracy:.2f}")

# Train the model on 5 epochs
num_epochs = 0
for i in range(num_epochs):
    print(f"Epoch: {i+1}/{num_epochs}")
    train()

#########################
### 6. Save the model ###
#########################
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}.")

MODEL_FILE = 'image-captioning-model.pt'
MODEL_BLOB = f'models/{MODEL_FILE}'
BUCKET_NAME = 'coms_6998_applied_ml'
torch.save(model, MODEL_FILE)
upload_to_gcs(BUCKET_NAME, MODEL_FILE, MODEL_BLOB)

# Also save the id_to_word and word_to_id maps to GCP
with open('id_to_word.json', 'w') as f:
    json.dump(id_to_word, f)
with open('word_to_id.json', 'w') as f:
    json.dump(word_to_id, f)
upload_to_gcs(BUCKET_NAME, 'id_to_word.json', 'models/id_to_word.json')
upload_to_gcs(BUCKET_NAME, 'word_to_id.json', 'models/word_to_id.json')

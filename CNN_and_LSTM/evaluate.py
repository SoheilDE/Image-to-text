import torch
from data_loader import get_loader
from cnn_model import CNN
from lstm_model import LSTM
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import json

# Define hyperparameters
batch_size = 32
embed_size = 256
hidden_size = 512
num_layers = 1

# Define transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define the device to evaluate on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data loader
val_loader = get_loader('./data/val/images/', './data/val/captions.json', batch_size, num_workers=2)

# Initialize the models
cnn = CNN(embed_size).to(device)
lstm = LSTM(embed_size, hidden_size, len(val_loader.dataset.vocab), num_layers).to(device)

# Load the trained weights
checkpoint = torch.load('./models/best-model.ckpt')
cnn.load_state_dict(checkpoint['cnn_state_dict'])
lstm.load_state_dict(checkpoint['lstm_state_dict'])

# Set the models to evaluation mode
cnn.eval()
lstm.eval()

# Define the references and hypotheses
references = []
hypotheses = []

# Generate captions for the validation images
with torch.no_grad():
    for i, (images, captions) in enumerate(val_loader):
        # Move batch of images to the GPU
        images = images.to(device)

        # Generate captions using beam search
        features = cnn(images)
        captions_pred = lstm.sample(features)

        # Convert captions_pred to a list of words
        captions_pred = captions_pred.tolist()
        captions_pred = [[val_loader.dataset.vocab.itos[idx] for idx in caption] for caption in captions_pred]

        # Convert the ground truth captions to a list of words
        captions = captions.tolist()
        captions = [[val_loader.dataset.vocab.itos[idx] for idx in caption if idx != 0] for caption in captions]

        # Append the references and hypotheses
        references.extend(captions)
        hypotheses.extend(captions_pred)

# Compute the BLEU score
bleu_score = corpus_bleu([[ref] for ref in references], hypotheses)

# Print the BLEU score
print('BLEU-4 score: {:.4f}'.format(bleu_score))

# Save the references and hypotheses to a JSON file
with open('./results/references.json', 'w') as f:
    json.dump(references, f)
with open('./results/hypotheses.json', 'w') as f:
    json.dump(hypotheses, f)
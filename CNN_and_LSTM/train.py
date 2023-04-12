import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loader
from cnn_model import CNN
from lstm_model import LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Define hyperparameters
batch_size = 32
embed_size = 256
hidden_size = 512
num_layers = 1
learning_rate = 0.001
num_epochs = 5

# Define transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Define the device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the data loaders
train_loader = get_loader('./data/train/images/', './data/train/captions.json', batch_size, num_workers=2)
val_loader = get_loader('./data/val/images/', './data/val/captions.json', batch_size, num_workers=2)

# Initialize the models and the optimizer
cnn = CNN(embed_size).to(device)
lstm = LSTM(embed_size, hidden_size, len(train_loader.dataset.vocab), num_layers).to(device)
optimizer = optim.Adam(list(cnn.fc.parameters()) + list(lstm.parameters()), lr=learning_rate)

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.vocab.stoi['<pad>'])

# Train the models
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, captions) in enumerate(train_loader):
        # Move batch of images and captions to the GPU
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        features = cnn(images)
        packed_captions = pack_padded_sequence(captions, batch_first=True,
                                               lengths=[len(caption) - 1 for caption in captions])
        outputs = lstm(features, packed_captions)
        loss = criterion(outputs.view(-1, len(train_loader.dataset.vocab)), captions[:, 1:].contiguous().view(-1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), torch.exp(loss.item())))

    # Save the model checkpoints
    torch.save({'epoch': epoch + 1,
                'cnn_state_dict': cnn.state_dict(),
                'lstm_state_dict': lstm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()}, './models/model-{}.ckpt'.format(epoch + 1))
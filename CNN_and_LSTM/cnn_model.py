import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, embed_size):
        super(CNN, self).__init__()

        # Load pretrained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        # Remove the last fully connected layer
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Add a new fully connected layer
        self.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.resnet.fc = self.fc

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features
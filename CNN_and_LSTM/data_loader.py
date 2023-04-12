import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json


class ImageCaptionDataset(Dataset):
    def __init__(self, images_path, captions_path, transform=None):
        self.images_path = images_path
        self.captions_path = captions_path
        self.transform = transform

        # Load captions
        with open(captions_path, 'r') as f:
            self.captions = json.load(f)

        # Preprocess captions
        self.captions = ['<start> ' + caption + ' <end>' for caption in self.captions]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_path = self.images_path + str(index) + '.jpg'
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, caption


def get_loader(images_path, captions_path, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = ImageCaptionDataset(images_path, captions_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader
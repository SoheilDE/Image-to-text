import torch
import torchvision.transforms as transforms
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class CaptionDataset(Dataset):
    def __init__(self, img_dir, caption_file, vocab, transform=None):
        self.img_dir = img_dir
        self.caption_file = caption_file
        self.vocab = vocab
        self.transform = transform

        with open(self.caption_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations[index]['image_id']
        caption = self.annotations[index]['caption']
        img = Image.open(os.path.join(self.img_dir, str(img_id)+'.jpg')).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        caption_encoded = [self.vocab['<SOS>']]
        caption_encoded += [self.vocab.get(word, self.vocab['<UNK>']) for word in caption.split()]
        caption_encoded += [self.vocab['<EOS>']]

        return img, torch.tensor(caption_encoded)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import CaptionDataset, collate_fn
from transformer_model import Transformer
from tqdm import tqdm
import os
import argparse
import json


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loader
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CaptionDataset(args.train_image_dir, args.train_caption_file, args.vocab, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    model = Transformer(args.embed_size, args.num_heads, args.num_layers, len(args.vocab), args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=args.vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.to(device)
            captions = captions.to(device)
            targets = captions[:, 1:]

            # Forward pass
            outputs = model(images, captions)
            outputs = outputs[:, :-1, :].contiguous().view(-1, len(args.vocab))
            targets = targets.contiguous().view(-1)

            # Compute loss and backpropagation
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training statistics
            if (i + 1) % args.print_every == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, args.num_epochs, i + 1, len(train_loader), loss.item()))

        # Save the model after every epoch
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.checkpoint_dir, 'epoch-{}.ckpt'.format(epoch + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, required=True, help='path for train images directory')
    parser.add_argument('--train_caption_file', type=str, required=True, help='path for train captions file')
    parser.add_argument('--val_image_dir', type=str, required=True, help='path for val images directory')
    parser.add_argument('--val_caption_file', type=str, required=True, help='path for val captions file')
    parser.add_argument('--vocab_file', type=str, required=True, help='path for vocab file')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='directory for saving checkpoints')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='number of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='path of checkpoint to resume training')
    args = parser.parse_args()

    train(args)
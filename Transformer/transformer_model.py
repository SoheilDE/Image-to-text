import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, dropout):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=2048, dropout=dropout)

        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions):
        embeddings = self.embedding(captions)
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.dropout(embeddings)

        # Reshape images to match the sequence length of captions
        images = images.unsqueeze(1)
        images = images.repeat(1, captions.size(1), 1, 1)

        # Concatenate image features and caption embeddings
        x = torch.cat((images, embeddings), dim=2)

        # Pass the concatenated tensor through the transformer
        x = self.transformer(x)

        # Apply linear layer to output of transformer
        x = self.fc(x)

        return x
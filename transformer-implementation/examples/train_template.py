#!/usr/bin/env python3
"""
Example Training Script Template
This is a template for training a complete Transformer model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO: Import your Encoder and Decoder classes
# from src.encoder import Encoder
# from src.decoder import Decoder


class Transformer(nn.Module):
    """Complete Transformer model"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1,
                 max_seq_length=200):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # TODO: Add positional encoding
        # self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # TODO: Add encoder and decoder
        # self.encoder = Encoder(d_model, num_heads, dropout, d_ff, num_layers)
        # self.decoder = Decoder(d_model, d_ff, num_heads, dropout, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and encode source
        src_embedded = self.src_embedding(src) * (self.d_model ** 0.5)
        # TODO: Add positional encoding
        # src_embedded = self.pos_encoding(src_embedded)
        
        # Embed target
        tgt_embedded = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        # TODO: Add positional encoding
        # tgt_embedded = self.pos_encoding(tgt_embedded)
        
        # Encode and decode
        # TODO: encoder_output = self.encoder(src_embedded, src_mask)
        # TODO: decoder_output = self.decoder(encoder_output, tgt_embedded, tgt_mask)
        
        # Project to vocabulary
        # TODO: output = self.output_projection(decoder_output)
        
        # return output
        pass


def train_step(model, src, tgt, optimizer, criterion):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # TODO: Implement training step
    # output = model(src, tgt[:, :-1])  # Teacher forcing
    # loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].view(-1))
    # loss.backward()
    # optimizer.step()
    
    # return loss.item()
    pass


def main():
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    
    # Initialize model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, 
                       num_heads, num_layers, d_ff, dropout)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding
    
    print("Model initialized. Ready for training!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()

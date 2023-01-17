import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int | None = None,
        padding_index: int = 0,
        pretrained_embedding: torch.Tensor | None = None, 
        freeze_embedding: bool = False,
        vocab_size: int | None = None,
        filter_sizes: list[int] = [3, 4, 5],
        num_filters: list[int] = [100, 100, 100],
        dropout: float = 0.5
    ):
        super(TextClassifier, self).__init__()

        #create embedding layer if pre-trained embedding is defined
        if pretrained_embedding is not None:
            self.vocab_size, self.embedding_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)

        #create embedding layer if pre-trained embedding is not defined
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, 
                embedding_dim=self.embedding_dim, 
                padding_idx=padding_index, 
                max_norm=5.0
            )

        #create convolutional network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embedding_dim, 
                out_channels=num_filters[i], 
                kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        #create fully-connected layer
        self.fully_connected = nn.Linear(np.sum(num_filters), num_classes)

        #create dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):   
        #get embeddings from 'input_ids'.
        x_embedding = self.embedding(input_ids).float() #output shape: (batch_size, max_len. embedding_dim)
        
        #permute 'x_embedding' to match input shape requirement of 'nn.Conv1d'
        x_reshaped = x_embedding.permute(0, 2, 1) #output shape: (batch_size, embedding_dim, max_len)

        #apply CNN and ReLU
        x_conv1d_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list] #output shape: (batch_size, num_filters[i], L_out)

        #apply max pooling
        x_pool_list = [F.max_pool1d(x_conv1d, kernel_size=x_conv1d.shape[2]) for x_conv1d in x_conv1d_list] #output shape: (batch_size, num_filters[i], 1)

        #concatenate x_pool_list to feed fully-connected layer
        x_fully_connected = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1) #output shape: (batch_size, sum(num_filters))

        #compute logits
        logits = self.fully_connected(self.dropout(x_fully_connected)) #output shape: (batch_size, num_classes)

        return logits
import torch
import torch.nn 
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dimension_model: int, vocab_size: int):
        super().__init__()
        self.dimension_model = dimension_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, dimension_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dimension_model)


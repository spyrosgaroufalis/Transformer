import torch
import torch.nn 
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dimension_model: int, vocab_size: int):
        super().__init__()
        self.dimension_model = dimension_model
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, dimension_model)

    # feed forward
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dimension_model)

class PositionalEncoding(nn.Module):
    def __init__(self, dimension_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dimension_model = dimension_model
        self.seq_len = seq_len
        self.dropout = torch.nn.Dropout(dropout)

        # matrix of (seq_len, dimension_model)
        pe = torch.zeros(seq_len, dimension_model)
        # creating the vector (tensor) of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension_model, 2).float() * (-math.log(10000.0) / dimension_model))
        # sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, dimension_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False) # dont want to learn the tensor
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(1)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

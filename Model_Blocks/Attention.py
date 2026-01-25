import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class Head(nn.Module):  # Cross-Attention Head if y is not None else Self-Attention Head
    def __init__(self,
            head_input_dim,
            head_size,
            head_output_dim
            ):
        super().__init__()
        self.Q = nn.Linear(head_input_dim, head_size, bias=False)
        self.K = nn.Linear(head_input_dim, head_size, bias=False)
        self.V = nn.Linear(head_input_dim, head_output_dim, bias=False)
        self.head_size = head_size

    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x
    
        q = self.Q(x) 
        k = self.K(y) 
        v = self.V(y)
    
        attention_scores = q @ k.transpose(1,2)* self.head_size**-0.5
    
        if mask is not None:
            attention_scores = attention_scores + mask  # Applying the mask (if any)
    
        attention_weights = F.softmax(attention_scores, dim=-1)
    
        context_vectors = attention_weights @ v
        return context_vectors


class MultiHeadAttention(nn.Module):
    def __init__(self,
            num_heads,
            head_input_dim,
            head_size,
            head_output_dim
            ):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_input_dim, head_size, head_output_dim) for _ in range(num_heads)])
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(num_heads * head_output_dim, head_input_dim) 

    def forward(self, x, y=None, mask=None):
        head_outputs = [head(x, y, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.linear(self.dropout(concatenated))
        return output

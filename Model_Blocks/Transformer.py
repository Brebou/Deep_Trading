import Model_Blocks.Encoder as enc
import torch.nn as nn
import torch

def positional_encoding(context_size, embedding_size):
    pe = torch.zeros(context_size, embedding_size)
    position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  
    return pe

class Transformer(nn.Module):
    def __init__(self,
        input_dim,  # nombre de features continues (156)
        num_classes,  # nombre de classes de sortie (26 stocks)
        embedding_size = 128,
        dropout_rate = 0.1,
        head_size = 64,
        num_heads = 4,
        n_encoder_blocks = 4,
        max_context_size = 64
        ):
        super().__init__()
        self.encoder = enc.Encoder(num_heads, embedding_size, head_size, embedding_size//num_heads, dropout_rate, n_encoder_blocks)
        self.positional_encoding = positional_encoding(max_context_size, embedding_size)
        
        # Projection des features continues vers l'espace d'embedding
        self.input_projection = nn.Linear(input_dim, embedding_size)
        
        # Classifier appliqué à chaque timestep
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_size, num_classes)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask_input=None):
        # x: [batch, seq_len, input_dim] avec des valeurs continues
        entry_embeddings = self.input_projection(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        entry_embeddings = self.dropout(entry_embeddings)
        
        # Encoder traite toute la séquence
        encoder_output = self.encoder(entry_embeddings, mask=mask_input)  # [batch, seq_len, embedding_size]
        
        # Classifier à chaque timestep
        output = self.classifier(encoder_output)  # [batch, seq_len, num_classes]
        
        return output
import Model_Blocks.Attention as att
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self,
            num_heads,
            head_input_dim,
            head_size,
            head_output_dim,
            embedding_size = 128,
            dropout_rate = 0.1
            ):
        super().__init__()
        self.multi_head_attention = att.MultiHeadAttention(num_heads, head_input_dim, head_size, head_output_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size,embedding_size*4),
            nn.Dropout(dropout_rate/2),
            nn.GELU(),
            nn.Linear(embedding_size*4, embedding_size)
        )
        self.layer_norm2 = nn.LayerNorm(head_input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate


    def forward(self, 
            x,
            mask=None
            ):
        
        multi_head_part = self.multi_head_attention(self.layer_norm1(x), mask=mask) # Self-Attention
        multi_head_part = self.dropout1(multi_head_part)
        x = x + multi_head_part  # First residual connection

        feed_forward_part = self.feed_forward(self.layer_norm2(x))
        feed_forward_part = self.dropout2(feed_forward_part)
        x = x + feed_forward_part  # Second residual connection
        

        return x
    
class Encoder(nn.Module):
    def __init__(self,
        num_heads,
        head_input_dim,
        head_size,
        head_output_dim,
        dropout_rate = 0.1,
        n_blocks = 2
        ):
        super().__init__()

        self.blocks = nn.ModuleList([EncoderBlock(num_heads,head_input_dim,head_size,head_output_dim,head_input_dim,dropout_rate) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(head_input_dim)

    def forward(self, x, mask):
        for layer in self.blocks: 
            x = layer(x, mask=mask)
        output = self.norm(x)
        return output

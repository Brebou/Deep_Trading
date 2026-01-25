import Model_Blocks.Attention as att
import torch.nn as nn
import torch


def create_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask

class DecoderBlock(nn.Module):
    def __init__(self,
        num_heads,
        embedding_size,
        head_size,
        dropout_rate = 0.1
        ):
        head_output_dim = embedding_size // num_heads
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.self_attention = att.MultiHeadAttention(num_heads,embedding_size,head_size,head_output_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.cross_attention = att.MultiHeadAttention(num_heads,embedding_size,head_size,head_output_dim)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size,embedding_size*4),
            nn.Dropout(dropout_rate/2),
            nn.GELU(),
            nn.Linear(embedding_size*4, embedding_size)
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self,
        x,
        y,
        cross_attention_mask = None
        ):
        # Self-Attention
        self_attention_mask = create_mask(x.size(1)).to(x.device) #Triangle mask for self-attention, to not see future tokens
        self_attention_part = self.self_attention(self.norm1(x), mask=self_attention_mask)
        self_attention_part = self.dropout1(self_attention_part)
        x = x + self_attention_part  # First residual connection

        # Cross-Attention
        cross_attention_part = self.cross_attention(self.norm2(x), y, mask=cross_attention_mask)
        cross_attention_part = self.dropout2(cross_attention_part)
        x = x + cross_attention_part  # Second residual connection

        # Feed-Forward
        feed_forward_part = self.feed_forward(self.norm3(x))
        feed_forward_part = self.dropout3(feed_forward_part)
        x = x + feed_forward_part  # Third residual connection
        return x


class Decoder(nn.Module):
    def __init__(self,
        num_heads,
        embedding_size,
        head_size,
        dropout_rate = 0.1,
        n_blocks = 2
        ):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(num_heads,embedding_size,head_size,dropout_rate) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self,
        x,
        y,
        cross_attention_mask = None
        ):
        for layer in self.blocks: 
            x = layer(x, y, cross_attention_mask=cross_attention_mask)
        return self.norm(x)
    
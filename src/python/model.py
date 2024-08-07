import torch
from torch import Tensor
import logging 

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")



class JointEmbedding(torch.nn.Module):
    """
    For a given token, its input representation is constructed by summing up
    the corresponding token, segment and positional embeddings. 
    This class combines the three embedding types. 
    """
    def __init__(self, vocab_size: int, dim: int, use_learnable_pos_embed: bool = False):
        super().__init__()
        self.dim = dim 
        self.token_embed = torch.nn.Embedding(vocab_size, dim)
        self.segment_embed = torch.nn.Embedding(vocab_size, dim)
        self.use_learnable_pos_embed = use_learnable_pos_embed
        if use_learnable_pos_embed:
            self.pos_embed = torch.nn.Embedding(vocab_size, dim)
        self.layer_norm = torch.nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        sentence_size = x.size(-1)
        position = self.positional_encoding(self.dim, x)

        segment_tensor = torch.zeros_like(x).to(device)
        segment_tensor[:, sentence_size // 2 + 1:] = 1

        output = self.token_embed(x)
        output += self.segment_embed(segment_tensor)
        if self.use_learnable_pos_embed:
            position = torch.arange(self.dim, dtype=torch.long).to(device)
            position = self.pos_embed(position.expand_as(x))
        else:
            position = self.positional_encoding(self.dim, x)
        output += position 

        return self.layer_norm(output)

    def positional_encoding(self, dim: int, x: Tensor) -> Tensor:
        batch_size, stce_size = x.size()
        position = torch.arange(stce_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = 2 * d / dim 

        position = position.unsqueeze(1)
        position = position / (1e4 ** d)
        position[:, 0::2] = torch.sin(position[:, 0::2])
        position[:, 1::2] = torch.cos(position[:, 1::2])

        return position.expand(batch_size, *position.size())
    

class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.key_linear = torch.nn.Linear(dim_in, dim_out)
        self.query_linear = torch.nn.Linear(dim_in, dim_out)
        self.value_linear = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """
        x: The output of the JointEmbedding module
        attention_mask: The vector that masks out the [PAD] tokens
        
        x.shape: B, T, D where B: batch size, T: token seq length and D: dimension
        attention_mask.shape: B, 1, D
        """
        Q, K, V = self.query_linear(x), self.key_linear(x), self.value_linear(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) * Q.size(1) ** -0.5

        scores = scores.masked_fill_(attention_mask, 1e-9)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)
        return context 
    

        


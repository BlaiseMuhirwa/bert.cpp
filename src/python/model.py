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

    def __init__(
        self, vocab_size: int, dim: int, use_learnable_pos_embed: bool = False
    ):
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
        segment_tensor[:, sentence_size // 2 + 1 :] = 1

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
        position = position / (1e4**d)
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

        x.shape: B, T, D_in where B: batch size, T: token seq length and D_in: input dimension
        attention_mask.shape: B, 1, T
        """
        # Q, K and V: All have shape (B, T, D_out)
        Q, K, V = self.query_linear(x), self.key_linear(x), self.value_linear(x)
        # Scores have shape (B, T, T)
        scores = torch.bmm(Q, K.transpose(1, 2)) * Q.size(1) ** -0.5

        scores = scores.masked_fill_(attention_mask, 1e-9)
        # attn has shape (B, T, T)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        # context has shape (B, T, D_out)
        context = torch.bmm(attn, V)
        return context


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads: int, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(dim_in=dim_in, dim_out=dim_out) for _ in range(n_heads)]
        )
        self.linear = torch.nn.Linear(dim_out * n_heads, dim_in)
        self.layer_norm = torch.nn.LayerNorm(dim_in)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        outputs = [head(x, attention_mask) for head in self.heads]
        scores = torch.cat(outputs, dim=-1)
        scores = self.layer_norm(self.linear(scores))
        return scores


class Encoder(torch.nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, attn_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_heads=attn_heads, dim_in=dim_in, dim_out=dim_out
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.Dropout(dropout),
            torch.nn.GELU(),
            torch.nn.Linear(dim_out, dim_in),
            torch.nn.Dropout(dropout),
        )
        self.layer_norm = torch.nn.LayerNorm(dim_in)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        context = self.attn(x, attention_mask)
        return self.layer_norm(self.ffn(context))


class Bert(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_in: int,
        dim_out: int,
        attn_heads: int = 4,
        dropout: float = 0.1,
        use_learnable_pos_embed: bool = False,
    ):
        super().__init__()
        self.embed = JointEmbedding(
            vocab_size=vocab_size,
            dim=dim_in,
            use_learnable_pos_embed=use_learnable_pos_embed,
        )
        self.encoder = Encoder(
            dim_in=dim_in, dim_out=dim_out, attn_heads=attn_heads, dropout=dropout
        )

        # Massive layer for token prediction task
        self.proj = torch.nn.Linear(dim_in, vocab_size)

        # This is for the next sentence prediction task.
        self.cls_layer = torch.nn.Linear(dim_in, 2)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        embedding = self.embed(x)
        encoded = self.encoder(embedding, attention_mask)
        token_predictions = self.proj(encoded)

        first_word = encoded[:, 0, :]
        logits = torch.nn.functional.log_softmax(token_predictions, dim=-1)
        cls_output = self.cls_layer(first_word)
        return logits, cls_output

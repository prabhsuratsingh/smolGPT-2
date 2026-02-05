import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size should be divisible by number of heads"

        self.SEQ = nn.Linear(embed_size, embed_size * 3, bias=True)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, seq, mask):
        N, seq_len, _ = seq.shape
        
        QKV = self.SEQ(seq)
        QKV = QKV.reshape(N, seq_len, 3, self.heads, self.head_dim)
        QKV = QKV.permute(2, 0, 3, 1, 4)

        Q, K, V = QKV[0], QKV[1], QKV[2]

        energy = torch.einsum("nhqd,nhkd->nhqk", [Q, K])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, torch.finfo(energy.dtype).min)

        #original gpt applies dropout here too, add afterwards
        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)

        out = torch.einsum("nhqk,nhkd->nhqd", [attention, V])
        out = out.permute(0, 2, 1, 3).reshape(N, seq_len, self.embed_size)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, mask):

        x = x + self.dropout(
            self.attention(self.norm1(x), mask)
        )

        x = x + self.dropout(
            self.feed_forward(self.norm2(x))
        )

        return x
    
class Decoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            embed_size,
            num_layers,
            heads, 
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)


        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion, device)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.fc_out.weight = self.word_embedding.weight


    def forward(self, x, target_mask):
        N, seq_len = x.shape

        pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        pos = pos.expand(N, seq_len)
        
        x = self.dropout(
            self.word_embedding(x) + self.position_embedding(pos)
        )

        for layer in self.layers:
            x = layer(x, target_mask)

        ln = self.dropout(self.norm(x))
        out = self.fc_out(ln)

        return out

class GPT2(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            src_pad_index,
            target_pad_index,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100
    ):
        super(GPT2, self).__init__()

        self.decoder = Decoder(
            src_vocab_size,
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_index = src_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def make_padding_mask(self, target):
        # target: (N, T)
        return (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)
        # shape: (N, 1, 1, T)


    def make_target_mask(self, target):
        N, T = target.shape

        causal_mask = torch.tril(torch.ones((T, T), device=target.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        # (1, 1, T, T)

        padding_mask = (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, T)

        return causal_mask * padding_mask

    
    def forward(self, target):
        target_mask = self.make_target_mask(target)

        out = self.decoder(target, target_mask)

        return out
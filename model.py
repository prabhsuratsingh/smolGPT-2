import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bpe import BPE

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size should be divisible by number of heads"

        self.SEQ = nn.Linear(embed_size, embed_size * 3, bias=True)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, seq, mask):
        N, seq_len, _ = seq.shape
        
        QKV = self.SEQ(seq)
        QKV = QKV.reshape(N, seq_len, 3, self.heads, self.head_dim)
        QKV = QKV.permute(2, 0, 3, 1, 4)

        Q, K, V = QKV[0], QKV[1], QKV[2]

        energy = torch.einsum("nhqd,nhkd->nhqk", [Q, K])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, torch.finfo(energy.dtype).min)

        attention = self.attn_dropout(
            torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)
        )

        out = torch.einsum("nhqk,nhkd->nhqd", [attention, V])
        out = out.permute(0, 2, 1, 3).reshape(N, seq_len, self.embed_size)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads, dropout=dropout)
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
    
class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.tokens = tokenizer.encode(
            text,
            allowed_special={"<|endoftext|>"}
        )

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(
            self.tokens[idx : idx + self.block_size],
            dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[idx + 1 : idx + self.block_size + 1],
            dtype=torch.long
        )
        return x, y
    
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -model.decoder.position_embedding.num_embeddings:]
        logits = model(tokens_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPE()
    tokenizer.load_vocab_and_merges(vocab_path="vocab.json", bpe_merges_path="bpe_merges.txt")

    vocab_size = len(tokenizer.vocab)

    model = GPT2(
        src_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        src_pad_index=tokenizer.get_special_token_id("<|endoftext|>"),
        target_pad_index=tokenizer.get_special_token_id("<|endoftext|>"),
        embed_size=256,
        num_layers=6,
        heads=8,
        max_length=128
    ).to(device)

    dataset = GPT2Dataset(
        text=text,
        tokenizer=tokenizer,
        block_size=128
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 5
    log_interval = 100 
    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        running_loss = 0.0

        for step, (x, y) in enumerate(loader):
            global_step += 1
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            num_batches += 1

            if global_step % log_interval == 0:
                avg_step_loss = running_loss / log_interval
                print(
                    f"step {global_step:6d} | "
                    f"epoch {epoch+1} | "
                    f"loss {avg_step_loss:.4f}"
                )
                running_loss = 0.0

        avg_epoch_loss = epoch_loss / num_batches
        print(
            f"epoch {epoch+1}/{num_epochs} | "
            f"avg loss {avg_epoch_loss:.4f}"
        )

    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    torch.save(model.state_dict(), "gpt2_shakespeare.pt")
    print("Model saved.")

    print("Test Generation : ")
    print(generate(model, tokenizer, "Once upon a time"))
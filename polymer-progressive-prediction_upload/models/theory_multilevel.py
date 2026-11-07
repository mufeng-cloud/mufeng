from utils.theory_multilevel import *

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, 7, padding=3)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        res = x
        out = self.drop(self.act(self.bn1(x)))
        out = self.conv1(out)
        out = self.drop(self.act(self.bn2(out)))
        out = self.conv2(out)
        return out + res

class PolymerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        self.shared = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.PReLU(),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.PReLU(),
        )
        self.fusion = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.branch = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
        )
        self.head_re = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(init_weights)

    def forward(self, x1, x2, x3, use_ch3):
        e1 = self.embed(x1).transpose(1, 2)
        e2 = self.embed(x2).transpose(1, 2)
        e3 = self.embed(x3).transpose(1, 2)

        f1 = self.shared(e1)
        f2 = self.shared(e2)
        f3 = self.shared(e3)

        fused12 = self.cnn(f1 + f2)
        mask = use_ch3.float().view(-1, 1, 1)
        fused = fused12 + f3 * mask

        fused = self.fusion(fused)
        h = self.pool(fused).squeeze(-1)
        h = self.branch(h)

        return self.head_re(h).squeeze(-1)

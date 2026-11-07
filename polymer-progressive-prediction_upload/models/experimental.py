from utils.experimental import *

class ResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + residual

class PolymerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        self.shared_extractor = nn.Sequential(
            ResNetBlock(embed_dim),
            ResNetBlock(embed_dim),
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.PReLU(),

            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(embed_dim),
            nn.PReLU(),
        )

        self.fusion_net = nn.Sequential(
            ResNetBlock(embed_dim),
            ResNetBlock(embed_dim),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        def make_head():
            return nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, 1)
            )

        self.head_mw  = make_head()
        self.head_eta = make_head()
        self.head_tg  = make_head()

    def forward(self, x1, x2, x3, use_ch3):
        e1 = self.embed(x1).transpose(1, 2)
        e2 = self.embed(x2).transpose(1, 2)
        e3 = self.embed(x3).transpose(1, 2)

        f1 = self.shared_extractor(e1)
        f2 = self.shared_extractor(e2)
        f3 = self.shared_extractor(e3)

        fused12 = f1 + f2
        fused12 = self.cnn(fused12)

        mask = use_ch3.float().view(-1, 1, 1)
        fused = fused12 + f3 * mask

        fused = self.fusion_net(fused)
        pooled = self.global_pool(fused).squeeze(-1)

        return (
            self.head_mw(pooled).squeeze(-1),
            self.head_eta(pooled).squeeze(-1),
            self.head_tg(pooled).squeeze(-1)
        )

from utils.theory_progressive_ddp import *

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
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

class HierarchicalDigitHeadIndependent(nn.Module):
    """Stepwise digit prediction: 万→千→百→十→个, each stage conditioned on true/prev digit."""

    def __init__(self, hidden_dim: int = 128, digit_dim: int = 8):
        super().__init__()
        self.digit_emb = nn.Embedding(10, digit_dim)

        self.stage1 = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.PReLU(),
            nn.Linear(64, 16), nn.PReLU(),
            nn.Linear(16, 1),  # ten-thousands digit
        )
        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim + digit_dim, 32), nn.PReLU(),
            nn.Linear(32, 1),  # thousands digit
        )
        self.stage3_hundreds = nn.Linear(hidden_dim + digit_dim, 1)
        self.stage4_tens = nn.Linear(hidden_dim + digit_dim, 1)
        self.stage5_ones = nn.Linear(hidden_dim + digit_dim, 1)

    def forward(self, h, target=None):
        if target is not None:  # teacher-forcing (training)
            t = target.long()
            d1 = (t // 10000).clamp(0, 9)
            d2 = ((t // 1000) % 10).clamp(0, 9)
            d3 = ((t // 100) % 10).clamp(0, 9)
            d4 = ((t // 10) % 10).clamp(0, 9)
            d5 = (t % 10).clamp(0, 9)

            p1 = self.stage1(h).squeeze(-1)
            p2 = self.stage2(torch.cat([h, self.digit_emb(d1)], dim=1)).squeeze(-1)
            p3 = self.stage3_hundreds(torch.cat([h, self.digit_emb(d2)], dim=1)).squeeze(-1)
            p4 = self.stage4_tens(torch.cat([h, self.digit_emb(d3)], dim=1)).squeeze(-1)
            p5 = self.stage5_ones(torch.cat([h, self.digit_emb(d4)], dim=1)).squeeze(-1)

            value = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
            return value.float(), [p1, p2, p3, p4, p5]

        else:  # autoregressive inference
            d1_raw = self.stage1(h).squeeze(-1)
            d1 = d1_raw.round().clamp(0, 9).long()

            d2_raw = self.stage2(torch.cat([h, self.digit_emb(d1)], dim=1)).squeeze(-1)
            d2 = d2_raw.round().clamp(0, 9).long()

            d3_raw = self.stage3_hundreds(torch.cat([h, self.digit_emb(d2)], dim=1)).squeeze(-1)
            d3 = d3_raw.round().clamp(0, 9).long()

            d4_raw = self.stage4_tens(torch.cat([h, self.digit_emb(d3)], dim=1)).squeeze(-1)
            d4 = d4_raw.round().clamp(0, 9).long()

            d5_raw = self.stage5_ones(torch.cat([h, self.digit_emb(d4)], dim=1)).squeeze(-1)
            d5 = d5_raw.round().clamp(0, 9).long()

            value = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
            return value.float(), None

class PolymerModel(nn.Module):
    """Three-channel backbone + stepwise digit head."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, digit_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.shared = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.PReLU(),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.PReLU(),
        )
        self.fusion = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.branch = nn.Linear(embed_dim, hidden_dim)
        self.head_re = HierarchicalDigitHeadIndependent(hidden_dim=hidden_dim, digit_dim=digit_dim)
        self.apply(init_weights)

    def forward(self, x1, x2, x3, use_ch3, target_int=None):
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

        return self.head_re(h, target_int)

def hierarchical_digit_loss(preds_list, targets):
    """Weighted MSE over 5 digits. 万位权重大, 个位权重小."""
    t = targets.long()
    d1 = (t // 10000).float()
    d2 = ((t // 1000) % 10).float()
    d3 = ((t // 100) % 10).float()
    d4 = ((t // 10) % 10).float()
    d5 = (t % 10).float()

    return (
        150 * F.mse_loss(preds_list[0], d1)
        + 120 * F.mse_loss(preds_list[1], d2)
        + 20 * F.mse_loss(preds_list[2], d3)
        + 10 * F.mse_loss(preds_list[3], d4)
        + 10 * F.mse_loss(preds_list[4], d5)
    )

from utils.theory_cnn import *

class CNN_model(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dp_hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.dp_mlp = nn.Sequential(
            nn.Linear(1, dp_hidden),
            nn.ReLU(),
            nn.Linear(dp_hidden, dp_hidden),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + dp_hidden, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        self.head_re = nn.Linear(hidden_dim, 1)

    def forward(self, x, dp):
        e = self.embed(x).transpose(1, 2)
        h_struct = self.cnn(e).flatten(1)

        h_dp = self.dp_mlp(dp)

        h = torch.cat([h_struct, h_dp], dim=1)
        h = self.fusion(h)

        return self.head_re(h).squeeze(-1)

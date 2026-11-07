from utils.theory_progressive import *

class HierarchicalDigitHeadIndependent(nn.Module):
    def __init__(self, hidden_dim=128, digit_dim=8):
        super().__init__()
        self.digit_emb = nn.Embedding(10, digit_dim)

        self.stage1 = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.PReLU(),          
             nn.Linear(64, 16), nn.PReLU(), 
          nn.Linear(16, 1)
        )

        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim+ digit_dim, 32), nn.PReLU(),
             nn.Linear(32, 1)
        )
        self.stage3_hundreds =  nn.Linear(hidden_dim + digit_dim, 1)                      


        self.stage4_tens = nn.Linear(hidden_dim + digit_dim, 1)
              
          

        self.stage5_ones = nn.Linear(hidden_dim + digit_dim, 1)
            
      
    def forward(self, h, target=None):
        if target is not None:
            true = target.long()

            d1_true = (true // 10000).clamp(0, 9)
            d2_true = ((true // 1000) % 10).clamp(0, 9)
            d3_true = ((true // 100) % 10).clamp(0, 9)
            d4_true = ((true // 10) % 10).clamp(0, 9)
            d5_true = (true % 10).clamp(0, 9)

            d1_pred = self.stage1(h).squeeze(-1)

            cond2 = torch.cat([h, self.digit_emb(d1_true)], dim=1)
            d2_pred = self.stage2(cond2).squeeze(-1)

            cond3 = torch.cat([h, self.digit_emb(d2_true)], dim=1)
            d3_pred = self.stage3_hundreds(cond3).squeeze(-1)

            cond4 = torch.cat([h, self.digit_emb(d3_true)], dim=1)
            d4_pred = self.stage4_tens(cond4).squeeze(-1)

            cond5 = torch.cat([h, self.digit_emb(d4_true)], dim=1)
            d5_pred = self.stage5_ones(cond5).squeeze(-1)

            value = d1_true * 10000 + d2_true * 1000 + d3_true * 100 + d4_true * 10 + d5_true
            return value.float(), [d1_pred, d2_pred, d3_pred, d4_pred, d5_pred]

        else:

            d1_pred = self.stage1(h).squeeze(-1)
            d1 = d1_pred.round().clamp(0, 9).long()

            cond2 = torch.cat([h, self.digit_emb(d1)], dim=1)
            d2_pred = self.stage2(cond2).squeeze(-1)
            d2 = d2_pred.round().clamp(0, 9).long()

            cond3 = torch.cat([h, self.digit_emb(d2)], dim=1)
            d3_pred = self.stage3_hundreds(cond3).squeeze(-1)
            d3 = d3_pred.round().clamp(0, 9).long()

            cond4 = torch.cat([h, self.digit_emb(d3)], dim=1)
            d4_pred = self.stage4_tens(cond4).squeeze(-1)
            d4 = d4_pred.round().clamp(0, 9).long()

            cond5 = torch.cat([h, self.digit_emb(d4)], dim=1)
            d5_pred = self.stage5_ones(cond5).squeeze(-1)
            d5 = d5_pred.round().clamp(0, 9).long()

            value = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
            return value.float(), None

def hierarchical_digit_loss_independent(preds_list, targets):
    true = targets.long()

    d1_true = (true // 10000).float()
    d2_true = ((true // 1000) % 10).float()
    d3_true = ((true // 100) % 10).float()
    d4_true = ((true // 10) % 10).float()
    d5_true = (true % 10).float()

    L1 = F.mse_loss(preds_list[0], d1_true)
    L2 = F.mse_loss(preds_list[1], d2_true)
    L3 = F.mse_loss(preds_list[2], d3_true)
    L4 = F.mse_loss(preds_list[3], d4_true)
    L5 = F.mse_loss(preds_list[4], d5_true)
    
    return 150 * L1 + 120 * L2 + 20 * L3 + 10 * L4 + 10 * L5

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
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
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, digit_dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)

        self.shared = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.PReLU(),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.PReLU(),
        )
        self.fusion = nn.Sequential(ResBlock(embed_dim), ResBlock(embed_dim))
        self.pool = nn.AdaptiveAvgPool1d(1)

        #self.branch = nn.Sequential(
         #   nn.Linear(embed_dim, hidden_dim), nn.PReLU(),
         #   nn.Linear(hidden_dim, hidden_dim), nn.PReLU(),
       
       # )
        
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

import torch
from torch import nn
from configs.public_tg_protocol import DIGIT_WEIGHTS

def encode_digits(values):
    v=values.reshape(-1)
    if not torch.isfinite(v).all() or (v<0).any():raise ValueError('Invalid log10 targets')
    grid=torch.round(v.double()*10000).long()
    if not torch.allclose(v.double(),grid.double()/10000,atol=5e-7,rtol=0):raise ValueError('Expected original CSV four-decimal log10 labels')
    q=torch.div(grid,10,rounding_mode='floor')
    if (q>9999).any():raise ValueError('Scaled target does not fit four digits')
    return q,torch.stack((q//1000,q//100%10,q//10%10),1)

def hard_digits(v):return torch.floor(v+.5).clamp(0,9).long()
def reconstruct_log(raw,hard=False):
    digits=hard_digits(raw) if hard else raw
    return (digits*raw.new_tensor([1000,100,10])).sum(1,keepdim=True)/1000

def ste_reconstruct_log(raw):
    hard=hard_digits(raw).to(raw.dtype)
    ste_digits=raw+(hard-raw).detach()
    return (ste_digits*raw.new_tensor([1000,100,10])).sum(1,keepdim=True)/1000

def weighted_digit_mse(raw,targets):
    return ((raw-targets.to(raw.dtype)).square().mean(0)*raw.new_tensor(DIGIT_WEIGHTS)).sum()


class NumericDigitEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(1,16),nn.PReLU(),nn.Linear(16,16))
    def forward(self,d):return self.layers(d.to(self.layers[0].weight.dtype).unsqueeze(-1)/9)

class ProgressiveTgHead(nn.Module):
    def __init__(self,dropout=0.):
        super().__init__()
        self.thousands_dropout=nn.Dropout(dropout);self.hundreds_dropout=nn.Dropout(dropout);self.tens_dropout=nn.Dropout(dropout)
        self.embedding=NumericDigitEncoder()
        self.thousands=nn.Linear(300,1)
        self.hundreds=nn.Sequential(nn.Linear(316,64),nn.PReLU(),nn.Linear(64,1))
        self.tens=nn.Sequential(nn.Linear(316,16),nn.PReLU(),nn.Linear(16,1))
    def forward(self,hidden,targets=None):
        if self.training and targets is None:raise ValueError('Training requires true previous-digit targets')
        if not self.training and targets is not None:raise ValueError('Inference must not receive target conditions')
        truth=encode_digits(targets)[1] if targets is not None else None
        outputs=[]
        for i,name in enumerate(['thousands','hundreds','tens']):
            x=hidden
            if i:
                d=truth[:,i-1] if truth is not None else hard_digits(outputs[-1])
                x=torch.cat((hidden,self.embedding(d)),1)
            raw=9*torch.sigmoid(getattr(self,name)(getattr(self,name+'_dropout')(x))).squeeze(-1)
            outputs.append(raw)
        raw=torch.stack(outputs,1)
        return raw,reconstruct_log(raw,hard=True)

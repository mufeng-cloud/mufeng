from utils.theory_ablation import *

class CharTokenizer:
    def __init__(self):
        self.char2id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
        self.id2char = {PAD_ID: "<PAD>", UNK_ID: "<UNK>"}

    def add_char(self, ch: str):
        if ch not in self.char2id:
            idx = len(self.char2id)
            self.char2id[ch] = idx
            self.id2char[idx] = ch

    def build_from_texts(self, texts: List[str]):
        for t in texts:
            if not isinstance(t, str): continue
            for ch in t:
                self.add_char(ch)

    def encode(self, text: Optional[str], max_len: int) -> List[int]:
        if not text or not isinstance(text, str):
            return [PAD_ID] * max_len
        ids = [self.char2id.get(ch, UNK_ID) for ch in text[:max_len]]
        return ids + [PAD_ID] * (max_len - len(ids))

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

def build_channel2(df: pd.DataFrame) -> List[str]:
    seqs = []
    for _, row in df.iterrows():
        N = int(row["num_unit"])
        seqs.append('A' * min(N, CH2_MAX_LEN))
    return seqs

class TargetScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std  = float(std) if std > 0 else 1.0
    def encode(self, x): return (x - self.mean) / self.std
    def decode(self, z): return z * self.std + self.mean

class PolymerDataset_Ablation(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CharTokenizer, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = str(row["smiles"])
        s1 = f"A:{smiles}"
        s2 = str(row["seq_ch2"])

        x1 = torch.tensor(self.tok.encode(s1, CH1_MAX_LEN), dtype=torch.long)
        x2 = torch.tensor(self.tok.encode(s2, CH2_MAX_LEN), dtype=torch.long)

        y_re = torch.tensor([row["Re"]], dtype=torch.float32)

        return x1, x2, y_re

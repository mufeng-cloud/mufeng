from utils.theory_multilevel import *

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
            if not isinstance(t, str):
                continue
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

def build_channel23_sequences(df: pd.DataFrame,
                              ch2_max_len: int = CH2_MAX_LEN,
                              ch3_group: int = 30) -> Tuple[List[str], List[Optional[str]], List[bool]]:
    ch2_list, ch3_list, use_ch3_flags = [], [], []
    for _, row in df.iterrows():
        N = int(row["num_unit"])
        if N <= ch2_max_len:
            ch2_seq = 'A' * N
            ch3_seq = None
            use_ch3 = False
        else:
            ch2_seq = f"P:{'A' * ch3_group}"
            groups = int(math.ceil(N / ch3_group))
            ch3_seq = 'P' * groups
            use_ch3 = True
        ch2_list.append(ch2_seq)
        ch3_list.append(ch3_seq)
        use_ch3_flags.append(use_ch3)
    return ch2_list, ch3_list, use_ch3_flags

class TargetScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, z):
        return z * self.std + self.mean

class PolymerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CharTokenizer):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.df["use_ch3"] = self.df["use_ch3"].astype(bool)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = str(row["smiles"]) if "smiles" in row else ""
        s1 = f"A:{smiles}"
        s2 = str(row["seq_ch2"])
        s3 = row["seq_ch3"]

        x1 = torch.tensor(self.tok.encode(s1, CH1_MAX_LEN), dtype=torch.long)
        x2 = torch.tensor(self.tok.encode(s2, CH2_MAX_LEN), dtype=torch.long)
        x3 = torch.tensor(self.tok.encode(str(s3), CH3_MAX_LEN), dtype=torch.long) if s3 is not None and pd.notna(s3) else torch.tensor([PAD_ID] * CH3_MAX_LEN, dtype=torch.long)

        use_ch3 = torch.tensor(bool(row["use_ch3"]), dtype=torch.bool)
        y_re = torch.tensor([row["Re"]], dtype=torch.float32)

        return x1, x2, x3, use_ch3, y_re

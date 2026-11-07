from utils.experimental import *

class CharTokenizer:
    def __init__(self, add_chars: Optional[List[str]] = None):
        self.char2id = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
        self.id2char = {PAD_ID: "<PAD>", UNK_ID: "<UNK>"}
        if add_chars:
            for ch in add_chars:
                self.add_char(ch)

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
        if text is None or not isinstance(text, str):
            return [PAD_ID] * max_len
        ids = []
        for ch in text[:max_len]:
            ids.append(self.char2id.get(ch, UNK_ID))
        if len(ids) < max_len:
            ids += [PAD_ID] * (max_len - len(ids))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

def build_channel23_sequences(df: pd.DataFrame,
                              ch2_max_len: int,
                              ch3_max_len: int,
                              ch3_group: int = 10) -> Tuple[List[str], List[Optional[str]], List[bool]]:

    ch2_list, ch3_list, use_ch3_flags = [], [], []

    for _, row in df.iterrows():
        ru_mw = row.get("num_unit", np.nan)

        if pd.isna(ru_mw) or ru_mw <= 0:
            ch2_list.append("")
            ch3_list.append(None)
            use_ch3_flags.append(False)
            continue

        N = int(ru_mw)

        if N > ch2_max_len:
            ch2_seq = f"P:{'A' * ch3_group}"
            groups = int(math.ceil(N / float(ch3_group)))
            ch3_seq = 'P' * groups
            ch3_list.append(ch3_seq)
            use_ch3_flags.append(True)
        else:
            ch2_seq = 'A' * N
            ch3_list.append(None)
            use_ch3_flags.append(False)

        ch2_list.append(ch2_seq)

    return ch2_list, ch3_list, use_ch3_flags

@dataclass
class Scaler:
    mean: float
    std: float
    is_log: bool

    def encode(self, val: Optional[float]) -> Tuple[float, int]:
        if val is None or (isinstance(val, float) and (val != val)):
            return 0.0, 0
        v = float(val)
        return (v - self.mean) / (self.std if self.std > 0 else 1.0), 1

    def decode(self, val_norm: np.ndarray) -> np.ndarray:
        v = val_norm * (self.std if self.std > 0 else 1.0) + self.mean
        return v

class PolymerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CharTokenizer,
                 ch1_max_len: int, ch2_max_len: int, ch3_max_len: int,
                 scaler_dict: Dict[str, Scaler], mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.ch1_max_len = ch1_max_len
        self.ch2_max_len = ch2_max_len
        self.ch3_max_len = ch3_max_len
        self.scaler = scaler_dict
        self.mode = mode
        self.df["use_ch3"] = self.df["use_ch3"].astype(bool)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s1 = row.get("input_seq", "")
        s2 = row.get("seq_ch2", "")
        s3 = row.get("seq_ch3", None)
        use_ch3 = row["use_ch3"]

        x1 = torch.tensor(self.tok.encode(s1, self.ch1_max_len), dtype=torch.long)
        x2 = torch.tensor(self.tok.encode(s2, self.ch2_max_len), dtype=torch.long)
        if s3 is None:
            x3 = torch.tensor([PAD_ID] * self.ch3_max_len, dtype=torch.long)
        else:
            x3 = torch.tensor(self.tok.encode(s3, self.ch3_max_len), dtype=torch.long)

        y_mw, m_mw = self.scaler["Mw"].encode(row.get("Mw", None))
        y_eta, m_eta = self.scaler["viscosity"].encode(row.get("viscosity", None))
        y_tg, m_tg = self.scaler["Glass transition temperature"].encode(row.get("Glass transition temperature", None))

        if self.mode in ["train", "eval"]:
            return (x1, x2, x3,
                    torch.tensor(use_ch3, dtype=torch.bool),
                    torch.tensor([y_mw], dtype=torch.float32),
                    torch.tensor([y_eta], dtype=torch.float32),
                    torch.tensor([y_tg], dtype=torch.float32),
                    torch.tensor([m_mw], dtype=torch.long),
                    torch.tensor([m_eta], dtype=torch.long),
                    torch.tensor([m_tg], dtype=torch.long))

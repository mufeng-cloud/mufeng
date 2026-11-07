from utils.theory_cnn import *

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

class TargetScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    def encode(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / (self.std if self.std > 0 else 1.0)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z * (self.std if self.std > 0 else 1.0) + self.mean

class PolymerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CharTokenizer, max_len: int, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = row["polymer_seq"]
        x = torch.tensor(self.tok.encode(s, self.max_len), dtype=torch.long)

        dp = torch.tensor([float(row["num_unit"])], dtype=torch.float32)

        re = row["Re"]
        y_re = torch.tensor([float(re)], dtype=torch.float32)
        m_re = torch.tensor([1], dtype=torch.long)

        return x, dp, y_re, m_re

def fit_target_scaler(train_df: pd.DataFrame) -> TargetScaler:
    re_mask = (~train_df["Re"].isna())
    y_re = train_df.loc[re_mask, "Re"].values.astype(float)
    return TargetScaler(mean=float(np.mean(y_re)),
                        std=float(np.std(y_re)) if len(y_re) > 1 else 1.0)

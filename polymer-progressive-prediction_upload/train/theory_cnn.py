from utils.theory_cnn import *
from data.theory_cnn import *
from models.theory_cnn import *

def run_pipeline(
        out_dir: str = "results",
        max_len: int = 256,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 992603
):
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(seed)

    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

    if (len(train_df), len(val_df), len(test_df)) != (1574, 190, 196):
        raise ValueError("Paper dataset requires train=1574, val=190, test=196; refusing another split.")

    print(f"Loaded fixed dataset:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    scaler = fit_target_scaler(train_df)

    tok = CharTokenizer(add_chars=['{', '}'])
    tok.build_from_texts(train_df["polymer_seq"].tolist())

    tr_ds = PolymerDataset(train_df, tok, max_len, mode="train")
    va_ds = PolymerDataset(val_df, tok, max_len, mode="eval")
    te_ds = PolymerDataset(test_df, tok, max_len, mode="eval")

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = CNN_model(vocab_size=tok.vocab_size).to(device)
    model.apply(init_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tr_loader:
            x, dp, y_re_raw, m_re = [b.to(device) for b in batch]
            y_re_z = scaler.encode(y_re_raw.squeeze(-1).cpu().numpy())
            y_re_z = torch.from_numpy(y_re_z).to(device)

            pred_re_z = model(x, dp)
            mask_re = m_re.bool().squeeze(-1)
            loss = F.mse_loss(pred_re_z[mask_re], y_re_z[mask_re], reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch}/{epochs}] train_loss(z-space)={epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                x, dp, y_re_raw, m_re = [b.to(device) for b in batch]
                pred_re_z = model(x, dp)

                y_re_z = scaler.encode(y_re_raw.squeeze(-1).cpu().numpy())
                y_re_z = torch.from_numpy(y_re_z).to(device)

                mask_re = m_re.bool().squeeze(-1)
                val_loss += F.mse_loss(pred_re_z[mask_re], y_re_z[mask_re], reduction="sum").item()

        print(f"[Epoch {epoch}/{epochs}] val_loss(z-space)={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model with val_loss = {best_val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(out_dir, "best.pt"))

        scheduler.step()

    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds_re, trues_re = [], []

    with torch.no_grad():
        for batch in te_loader:
            x, dp, y_re_raw, m_re = [b.to(device) for b in batch]
            pre_z = model(x, dp)
            pre_raw = scaler.decode(pre_z.squeeze(-1).cpu().numpy())
            y_re_np = y_re_raw.squeeze(-1).cpu().numpy()
            m_re_np = m_re.squeeze(-1).cpu().numpy().astype(bool)

            preds_re.append(np.where(m_re_np, pre_raw, np.nan))
            trues_re.append(np.where(m_re_np, y_re_np, np.nan))

    trues_re = np.concatenate(trues_re)
    preds_re = np.concatenate(preds_re)
    smiles_test = test_df["smiles"].values  # preserved order (shuffle=False)

    combined_path = os.path.join(out_dir, "test_scatter_re")
    metrics = eval_and_plot(
        y_true=trues_re,
        y_pred=preds_re,
        smiles=smiles_test,
        path=combined_path
    )

    with open(os.path.join(out_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    paper_dir = os.path.join(out_dir, "figures_for_paper")
    os.makedirs(paper_dir, exist_ok=True)
    for file in glob.glob(os.path.join(out_dir, "test_scatter_re.*")):
        shutil.copy(file, paper_dir)

    print(f"\n所有投稿图已收集至：{paper_dir}")
    print("Test metrics:", metrics)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/theory_cnn"))
    parser.add_argument("--max_len", type=int, default=64, help="单通道最大序列长度")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(
        out_dir=args.out_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

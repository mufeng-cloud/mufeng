from utils.theory_multilevel import *
from data.theory_multilevel import *
from models.theory_multilevel import *

def run_pipeline(
    out_dir="re_only_final",
    seed=992603,
    epochs=120,
    batch_size=256,
    lr=5e-4
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

    if (len(train_df), len(val_df), len(test_df)) != (1574, 190, 196):
        raise ValueError("Paper dataset requires train=1574, val=190, test=196; refusing another split.")

    for df in [train_df, val_df, test_df]:
        df["seq_ch2"], df["seq_ch3"], df["use_ch3"] = build_channel23_sequences(df)

    print(f"Loaded fixed dataset:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tok = CharTokenizer()
    all_texts = (train_df["smiles"].apply(lambda x: f"A:{x}").tolist() +
                 train_df["seq_ch2"].tolist() +
                 [s for s in train_df["seq_ch3"].dropna() if s])
    tok.build_from_texts(all_texts)
    tok.add_char(':')
    tok.add_char('P')

    re_scaler = TargetScaler(train_df["Re"].mean(), train_df["Re"].std())

    tr_ds = PolymerDataset(train_df, tok)
    va_ds = PolymerDataset(val_df, tok)
    te_ds = PolymerDataset(test_df, tok)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

    model = PolymerModel(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-7)

    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        for x1, x2, x3, use_ch3, y_re in tr_loader:
            x1, x2, x3, use_ch3 = x1.to(device), x2.to(device), x3.to(device), use_ch3.to(device)
            y_re = y_re.to(device).squeeze(-1)

            re_z = re_scaler.encode(y_re)
            p_re = model(x1, x2, x3, use_ch3)
            loss = F.mse_loss(p_re, re_z)

            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2, x3, use_ch3, y_re in va_loader:
                x1, x2, x3, use_ch3 = (t.to(device) for t in [x1, x2, x3, use_ch3])
                y_re = y_re.to(device).squeeze(-1)
                re_z = re_scaler.encode(y_re)
                p_re = model(x1, x2, x3, use_ch3)
                val_loss += F.mse_loss(p_re, re_z, reduction='sum').item()

        print(f"Epoch {epoch:03d} | Val Loss (Re): {val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{out_dir}/best.pt")

    model.load_state_dict(torch.load(f"{out_dir}/best.pt", map_location=device))
    model.eval()
    preds_re, trues_re = [], []

    with torch.no_grad():
        for x1, x2, x3, use_ch3, y_re in te_loader:
            x1, x2, x3, use_ch3 = x1.to(device), x2.to(device), x3.to(device), use_ch3.to(device)
            p_re = model(x1, x2, x3, use_ch3)
            preds_re.append(re_scaler.decode(p_re.cpu().numpy()))
            trues_re.append(y_re.numpy())

    trues_re = np.concatenate(trues_re).flatten()
    preds_re = np.concatenate(preds_re)
    smiles_test = test_df["smiles"].values

    path = f"{out_dir}/re_scatter_test"
    metrics = eval_and_plot(trues_re, preds_re, smiles_test, path, task=r"$\langle R^2 \rangle_0$ (Å²)")

    json.dump(metrics, open(f"{out_dir}/metrics_test.json", "w"), indent=2)

    os.makedirs(f"{out_dir}/figures_for_paper", exist_ok=True)
    for ext in ["pdf", "png", "tiff"]:
        src = f"{path}.{ext}"
        if os.path.exists(src):
            shutil.copy(src, f"{out_dir}/figures_for_paper/")

    print("训练完成！结果已保存至:", out_dir)
    print(metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/theory_multilevel"))
    parser.add_argument("--seed", type=int, default=992603)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    run_pipeline(
        out_dir=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()

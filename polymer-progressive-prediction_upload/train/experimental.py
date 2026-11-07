from utils.experimental import *
from data.experimental import *
from models.experimental import *

def run_pipeline(
        data_csv: str,
        out_dir: str = "results",
        ch1_max_len: int = 256,
        ch2_max_len: int = 256,
        ch3_max_len: int = 256,
        ch3_group: int = 60,
        batch_size: int = 32,
        epochs: int = 100,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    df = pd.read_csv(data_csv)
    need_cols = ["polymer", "Glass transition temperature", "viscosity", "Mw", "num_unit"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")

    ch2_seq, ch3_seq, use_ch3_flags = build_channel23_sequences(df, ch2_max_len, ch3_max_len, ch3_group)
    df["input_seq"] = "A:" + df["polymer"].fillna("").astype(str)
    df["seq_ch2"] = ch2_seq
    df["seq_ch3"] = ch3_seq
    df["use_ch3"] = use_ch3_flags

    for i in range(min(5, len(df))):
        print(f"样本 {i}: num_unit={df.iloc[i]['num_unit']}, 通道2='{df.iloc[i]['seq_ch2']}', 通道3='{df.iloc[i]['seq_ch3']}', use_ch3={df.iloc[i]['use_ch3']}")

    split_plan = build_group_split_plan(df["polymer"].tolist(), seed=seed)
    save_group_split_plan(split_plan, data_csv, out_dir)
    test_df = df.iloc[split_plan["test_indices"]].reset_index(drop=True)
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    fold_summaries = []
    for fold, fold_split in enumerate(split_plan["folds"]):
        tr_df = df.iloc[fold_split["train_indices"]].reset_index(drop=True)
        va_df = df.iloc[fold_split["validation_indices"]].reset_index(drop=True)

        # Fit the character vocabulary on this training fold only.
        tok = CharTokenizer()
        tok.build_from_texts(tr_df["input_seq"].fillna("").tolist())
        tok.build_from_texts(tr_df["seq_ch2"].fillna("").tolist())
        tok.build_from_texts([s for s in tr_df["seq_ch3"].tolist() if isinstance(s, str)])
        tok.add_char(':')
        tok.add_char('P')
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        scalers: Dict[str, Scaler] = {}
        for task in ["Mw", "viscosity", "Glass transition temperature"]:
            col = tr_df[task]
            vals = col.dropna().values.astype(float)
            mean = float(np.mean(vals)) if len(vals) > 0 else 0.0
            std = float(np.std(vals)) if len(vals) > 0 else 1.0
            if std == 0.0:
                std = 1.0
            scalers[task] = Scaler(mean=mean, std=std, is_log=False)

        tr_ds = PolymerDataset(tr_df, tok, ch1_max_len, ch2_max_len, ch3_max_len, scalers, mode="train")
        va_ds = PolymerDataset(va_df, tok, ch1_max_len, ch2_max_len, ch3_max_len, scalers, mode="eval")
        te_ds = PolymerDataset(test_df, tok, ch1_max_len, ch2_max_len, ch3_max_len, scalers, mode="eval")

        model = PolymerModel(
            vocab_size=tok.vocab_size,
            embed_dim=128,
            hidden_dim=256,
        ).to(device)
        model.apply(init_weights)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False, worker_init_fn=worker_init_fn)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            task_losses = []
            for batch in tr_loader:
                x1, x2, x3, use_ch3, y_mw, y_eta, y_tg, m_mw, m_eta, m_tg = [b.to(device) for b in batch]
                y_mw = y_mw.squeeze(-1)
                y_eta = y_eta.squeeze(-1)
                y_tg = y_tg.squeeze(-1)
                m_mw = m_mw.squeeze(-1).bool()
                m_eta = m_eta.squeeze(-1).bool()
                m_tg = m_tg.squeeze(-1).bool()
                pred_mw, pred_eta, pred_tg = model(x1, x2, x3, use_ch3)
                batch_task_losses = []
                for pred, y, m, name in [
                    (pred_mw, y_mw, m_mw, "Mw"),
                    (pred_eta, y_eta, m_eta, "viscosity"),
                    (pred_tg, y_tg, m_tg, "Glass transition temperature"),
                ]:
                    if m.any():
                        L = F.mse_loss(pred[m], y[m], reduction="mean")
                        batch_task_losses.append(L)

                total_loss = sum(batch_task_losses) if batch_task_losses else torch.tensor(0.0, device=device)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                task_losses = batch_task_losses

            print(f"[Epoch {epoch}/{epochs}] train_loss={epoch_loss:.4f}")
            for i, name in enumerate(["Mw", "viscosity", "Glass transition temperature"]):
                if i < len(task_losses):
                    print(f"  {name} task loss: {task_losses[i].item():.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in va_loader:
                    x1, x2, x3, use_ch3, y_mw, y_eta, y_tg, m_mw, m_eta, m_tg = [b.to(device) for b in batch]
                    y_mw = y_mw.squeeze(-1)
                    y_eta = y_eta.squeeze(-1)
                    y_tg = y_tg.squeeze(-1)
                    m_mw = m_mw.squeeze(-1).bool()
                    m_eta = m_eta.squeeze(-1).bool()
                    m_tg = m_tg.squeeze(-1).bool()

                    pred_mw, pred_eta, pred_tg = model(x1, x2, x3, use_ch3)

                    for pred, y, m in [(pred_mw, y_mw, m_mw),
                                       (pred_eta, y_eta, m_eta),
                                       (pred_tg, y_tg, m_tg)]:
                        if m.any():
                            val_loss += F.mse_loss(pred[m], y[m], reduction="mean").item()

            print(f"[Epoch {epoch}] validation_loss={val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           os.path.join(fold_dir, "best.pt"))
            scheduler.step()

        ckpt = torch.load(os.path.join(fold_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()


        true_dict = {"Mw": [], "viscosity": [], "Glass transition temperature": []}
        pred_dict = {"Mw": [], "viscosity": [], "Glass transition temperature": []}

        with torch.no_grad():
            for batch in te_loader:
                x1, x2, x3, use_ch3, y_mw, y_eta, y_tg, m_mw, m_eta, m_tg = [b.to(device) for b in batch]
                pmw, peta, ptg = model(x1, x2, x3, use_ch3)

                for name, p, y, m, scaler in [
                    ("Mw", pmw, y_mw, m_mw, scalers["Mw"]),
                    ("viscosity", peta, y_eta, m_eta, scalers["viscosity"]),
                    ("Glass transition temperature", ptg, y_tg, m_tg, scalers["Glass transition temperature"]),
                ]:
                    p = p.squeeze(-1).cpu().numpy()
                    y = y.squeeze(-1).cpu().numpy()
                    m = m.squeeze(-1).cpu().numpy().astype(bool)

                    p_denorm = scaler.decode(p)
                    y_denorm = scaler.decode(y)

                    y_out = np.full_like(y_denorm, np.nan)
                    y_out[m] = y_denorm[m]

                    pred_dict[name].append(p_denorm)
                    true_dict[name].append(y_out)

        for name in true_dict:
            true_dict[name] = np.concatenate(true_dict[name])
            pred_dict[name] = np.concatenate(pred_dict[name])

        save_path = os.path.join(fold_dir, f"test_scatter_fold{fold}")
        metrics = eval_and_plot(true_dict, pred_dict, save_path, fold)


        figures_dir = os.path.join(fold_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        for ext in ["pdf", "png", "tiff"]:
            src = f"{save_path}.{ext}"
            if os.path.exists(src):
                shutil.copy(src, os.path.join(figures_dir, f"test_scatter_fold{fold}.{ext}"))

        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        fold_summaries.append(metrics)

    if fold_summaries:
        plot_cv_summary(fold_summaries, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=str(PROJECT_ROOT / "data/experimental/experimental_dataset.csv"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/experimental"))
    parser.add_argument("--ch1_max_len", type=int, default=256)
    parser.add_argument("--ch2_max_len", type=int, default=256)
    parser.add_argument("--ch3_max_len", type=int, default=256)
    parser.add_argument("--ch3_group", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=992603)
    args = parser.parse_args()

    set_seed(args.seed)
    run_pipeline(
        data_csv=args.data_csv,
        out_dir=args.out_dir,
        ch1_max_len=args.ch1_max_len,
        ch2_max_len=args.ch2_max_len,
        ch3_max_len=args.ch3_max_len,
        ch3_group=args.ch3_group,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed
    )

if __name__ == "__main__":
    main()

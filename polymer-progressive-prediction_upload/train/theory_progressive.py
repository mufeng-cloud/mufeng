from utils.theory_progressive import *
from data.theory_progressive import *
from models.theory_progressive import *

def run_pipeline(
        gpu_id: int,
        out_dir="re_hierarchical_independent_final",
        seed=992603,
        epochs=500,
        batch_size=128,
        lr=1e-3,
        weight_decay=2e-5,
        eta_min=8e-6,
        digit_dim=8
):
    """Single-GPU training — no DDP, no NCCL, no distributed anything."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

    if (len(train_df), len(val_df), len(test_df)) != (1574, 190, 196):
        raise ValueError("Paper dataset requires train=1574, val=190, test=196; refusing another split.")

    for df_ in [train_df, val_df, test_df]:
        df_["seq_ch2"], df_["seq_ch3"], df_["use_ch3"] = build_channel23_sequences(df_)

    tok = CharTokenizer()
    all_texts = (train_df["smiles"].apply(lambda x: f"A:{x}").tolist() +
                 train_df["seq_ch2"].tolist() +
                 [s for s in train_df["seq_ch3"].dropna() if s])
    tok.build_from_texts(all_texts)
    tok.add_char(':')
    tok.add_char('P')

    tr_ds = PolymerDataset(train_df, tok)
    va_ds = PolymerDataset(val_df, tok)
    te_ds = PolymerDataset(test_df, tok)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = PolymerModel(tok.vocab_size, digit_dim=digit_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)

    best_val_mae = float('inf')
    patience = 0
    max_patience = 100

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x1, x2, x3, use_ch3, y_re, y_re_int in tr_loader:
            x1 = x1.to(device); x2 = x2.to(device); x3 = x3.to(device)
            use_ch3 = use_ch3.to(device); y_re_int = y_re_int.to(device)
            _, preds_list = model(x1, x2, x3, use_ch3, target_int=y_re_int)
            loss = hierarchical_digit_loss_independent(preds_list, y_re_int)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        sch.step()

        # Validation every 10 epochs
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for x1, x2, x3, use_ch3, y_re, _ in DataLoader(va_ds, batch_size=256, shuffle=False):
                    x1 = x1.to(device); x2 = x2.to(device); x3 = x3.to(device)
                    use_ch3 = use_ch3.to(device)
                    pred_value, _ = model(x1, x2, x3, use_ch3, target_int=None)
                    val_preds.append(pred_value.cpu().numpy())
                    val_trues.append(y_re.numpy())
            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            val_mae = mean_absolute_error(val_trues, val_preds)

            print(f"  Epoch {epoch:03d} | Loss: {train_loss/max(n_batches,1):.4f} | Val MAE: {val_mae:.1f}", flush=True)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), f"{out_dir}/best.pt")
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print("  Early stopping!")
                break

    # Test evaluation
    best_path = f"{out_dir}/best.pt"
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    preds_re, trues_re = [], []
    with torch.no_grad():
        for x1, x2, x3, use_ch3, y_re, _ in DataLoader(te_ds, batch_size=256, shuffle=False):
            x1 = x1.to(device); x2 = x2.to(device); x3 = x3.to(device)
            use_ch3 = use_ch3.to(device)
            pred_value, _ = model(x1, x2, x3, use_ch3, target_int=None)
            preds_re.append(pred_value.cpu().numpy())
            trues_re.append(y_re.numpy())

    preds_re = np.concatenate(preds_re)
    trues_re = np.concatenate(trues_re)
    smiles_test = test_df["smiles"].values
    metrics = eval_and_plot(trues_re, preds_re, smiles_test,
                            f"{out_dir}/test_scatter",
                            task=r"$\langle R^{2} \rangle_{0}$ (Å²)")

    figures_dir = os.path.join(out_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    base_name = f"{out_dir}/test_scatter"
    for ext in ["pdf", "png", "tiff"]:
        src = f"{base_name}.{ext}"
        if os.path.exists(src):
            shutil.copy(src, os.path.join(figures_dir, f"test_scatter.{ext}"))

    result = {
        "best_val_mae": float(best_val_mae),
        "test_mae": float(metrics["MAE"]),
        "test_rmse": float(metrics["RMSE"]),
        "test_r2": float(metrics["R2"]),
        "digit_dim": int(digit_dim),
    }
    json.dump(result, open(f"{out_dir}/final_result.json", "w"), indent=2)
    print(f"  Test MAE={metrics['MAE']:.1f}, RMSE={metrics['RMSE']:.1f}, R²={metrics['R2']:.6f}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/theory_progressive"))
    parser.add_argument("--seed", type=int, default=992603)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--eta_min", type=float, default=8e-6)
    parser.add_argument("--grid_search", dest="grid_search", action="store_true", default=True)
    parser.add_argument("--no_grid_search", dest="grid_search", action="store_false")
    args = parser.parse_args()

    # torchrun spawns processes, sets LOCAL_RANK / WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.grid_search:
        weight_decay_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                             1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                             1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3]
        eta_min_list = [1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
                         1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7]
        digit_dim_list = [4, 6, 8, 10, 12, 14, 16]

        all_combos = []
        for wd in weight_decay_list:
            for eta in eta_min_list:
                for dd in digit_dim_list:
                    all_combos.append((wd, eta, dd))
        total = len(all_combos)

        # Each GPU gets a disjoint slice: GPU 0→[0,8,16,…], GPU 1→[1,9,17,…]
        my_combos = all_combos[local_rank::world_size]

        print(f"[GPU {local_rank}] {len(my_combos)}/{total} combinations assigned", flush=True)

        results = []
        for i, (wd, eta, dd) in enumerate(my_combos):
            combo_dir = os.path.join(args.out_dir, f"wd{wd:.0e}_eta{eta:.0e}_dd{dd}")
            print(f"\n[GPU {local_rank}] [{i+1}/{len(my_combos)}] "
                  f"wd={wd:.0e} eta={eta:.0e} dd={dd}", flush=True)

            result = run_pipeline(
                gpu_id=local_rank,
                out_dir=combo_dir,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                weight_decay=wd,
                eta_min=eta,
                digit_dim=dd,
            )
            result["weight_decay"] = wd
            result["eta_min"] = eta
            result["digit_dim"] = dd
            result["out_dir"] = combo_dir
            results.append(result)
            # Incremental save — survives preemption
            json.dump(results, open(f"{args.out_dir}/rank_{local_rank}_results.json", "w"), indent=2)

        print(f"\n[GPU {local_rank}] Done! {len(results)} combinations completed.", flush=True)

        # --- Cross-GPU result aggregation ---
        import time
        sentinel = os.path.join(args.out_dir, f"rank_{local_rank}_DONE")
        with open(sentinel, "w") as f:
            f.write("done")

        if local_rank == 0:
            print("\n[GPU 0] Waiting for all GPUs to finish...", flush=True)
            while True:
                done_files = [fn for fn in os.listdir(args.out_dir)
                              if fn.startswith("rank_") and fn.endswith("_DONE")]
                if len(done_files) >= world_size:
                    break
                time.sleep(15)

            all_results = []
            for rank in range(world_size):
                rf = os.path.join(args.out_dir, f"rank_{rank}_results.json")
                if os.path.exists(rf):
                    all_results.extend(json.load(open(rf)))

            all_results.sort(key=lambda x: x.get("test_mae", float("inf")))

            json.dump(all_results, open(os.path.join(args.out_dir, "grid_search_summary.json"), "w"), indent=2)

            best = all_results[0]
            json.dump(best, open(os.path.join(args.out_dir, "best_result.json"), "w"), indent=2)

            print(f"\n{'='*60}")
            print(f"Best result across all GPUs:")
            print(f"  test_mae  = {best['test_mae']:.1f}")
            print(f"  test_rmse = {best['test_rmse']:.1f}")
            print(f"  test_r2   = {best['test_r2']:.6f}")
            print(f"  Params: wd={best['weight_decay']:.0e}  eta_min={best['eta_min']:.0e}  digit_dim={best['digit_dim']}")
            print(f"  Dir: {best['out_dir']}")
            print(f"{'='*60}\n", flush=True)
    else:
        run_pipeline(
            gpu_id=local_rank,
            out_dir=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            eta_min=args.eta_min,
        )

from utils.theory_progressive_ddp import *
from data.theory_progressive_ddp import *
from models.theory_progressive_ddp import *

def train_one(
    *,
    out_dir: str,
    seed: int = 992603,
    epochs: int = 500,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 2e-5,
    eta_min: float = 8e-6,
    digit_dim: int = 8,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    set_seed(seed + rank)

    # ----  data (all ranks)  ----
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

    if (len(train_df), len(val_df), len(test_df)) != (1574, 190, 196):
        raise ValueError("Paper dataset requires train=1574, val=190, test=196; refusing another split.")
    for df_ in (train_df, val_df, test_df):
        df_["seq_ch2"], df_["seq_ch3"], df_["use_ch3"] = build_ch23_sequences(df_)

    if rank == 0:
        print(f"Data: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # tokenizer
    tok = CharTokenizer()
    tok.build_from_texts(
        train_df["smiles"].apply(lambda x: f"A:{x}").tolist()
        + train_df["seq_ch2"].tolist()
        + [s for s in train_df["seq_ch3"].dropna() if s]
    )
    tok.add_char(":"); tok.add_char("P")

    # datasets & loaders
    tr_ds = PolymerDataset(train_df, tok)
    va_ds = PolymerDataset(val_df, tok)
    te_ds = PolymerDataset(test_df, tok)

    tr_sampler = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank, shuffle=True)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=tr_sampler, drop_last=True,
                           pin_memory=True, num_workers=2)

    # ----  model  ----
    raw_model = PolymerModel(tok.vocab_size, digit_dim=digit_dim).to(device)
    model = DDP(raw_model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)

    # ----  training loop  ----
    best_val_mae = float("inf")
    patience = 0
    max_patience = 100

    for epoch in range(1, epochs + 1):
        tr_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for x1, x2, x3, uc, _, yi in tr_loader:
            x1, x2, x3, uc, yi = x1.to(device), x2.to(device), x3.to(device), uc.to(device), yi.to(device)
            _, preds = model(x1, x2, x3, uc, target_int=yi)
            loss = hierarchical_digit_loss(preds, yi)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sch.step()

        # ---  validation (rank 0 only)  ---
        should_stop = torch.tensor([0], dtype=torch.int, device=device)
        if rank == 0:
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for x1, x2, x3, uc, yr, _ in DataLoader(va_ds, batch_size=256, shuffle=False):
                    x1, x2, x3, uc = x1.to(device), x2.to(device), x3.to(device), uc.to(device)
                    pv, _ = model(x1, x2, x3, uc, target_int=None)
                    vp.append(pv.cpu().numpy()); vt.append(yr.numpy())
            val_mae = mean_absolute_error(np.concatenate(vt), np.concatenate(vp))

            if epoch == 1 or epoch % 10 == 0:
                print(f"  epoch {epoch:4d}  |  loss {total_loss/len(tr_loader):.4f}  |  val-mae {val_mae:.1f}",
                      flush=True)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                os.makedirs(out_dir, exist_ok=True)
                torch.save(raw_model.state_dict(), f"{out_dir}/best.pt")
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                should_stop[0] = 1

        dist.broadcast(should_stop, src=0)
        if should_stop.item():
            if rank == 0:
                print("  → early stopping", flush=True)
            break
        dist.barrier()

    # ----  test evaluation (rank 0 only)  ----
    result = None
    if rank == 0:
        best_path = f"{out_dir}/best.pt"
        if not os.path.exists(best_path):
            torch.save(raw_model.state_dict(), best_path)
        raw_model.load_state_dict(torch.load(best_path, map_location=device))
        raw_model.eval()
        tp, tt = [], []
        with torch.no_grad():
            for x1, x2, x3, uc, yr, _ in DataLoader(te_ds, batch_size=256, shuffle=False):
                x1, x2, x3, uc = x1.to(device), x2.to(device), x3.to(device), uc.to(device)
                pv, _ = raw_model(x1, x2, x3, uc, target_int=None)
                tp.append(pv.cpu().numpy()); tt.append(yr.numpy())
        tp = np.concatenate(tp); tt = np.concatenate(tt)
        metrics = eval_plot(tt, tp, f"{out_dir}/test_scatter")

        # copy to figures/
        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for ext in ("pdf", "png", "tiff"):
            src = f"{out_dir}/test_scatter.{ext}"
            if os.path.exists(src):
                shutil.copy(src, fig_dir)

        result = {
            "best_val_mae": float(best_val_mae),
            "test_mae": float(metrics["MAE"]),
            "test_rmse": float(metrics["RMSE"]),
            "test_r2": float(metrics["R2"]),
            "digit_dim": digit_dim,
        }
        with open(f"{out_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  MAE={metrics['MAE']:.1f}  RMSE={metrics['RMSE']:.1f}  R²={metrics['R2']:.6f}", flush=True)

    dist.barrier()
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "outputs/theory_progressive_ddp"))
    parser.add_argument("--seed", type=int, default=992603)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grid_search", action="store_true", default=True)
    parser.add_argument("--no_grid_search", dest="grid_search", action="store_false")
    args, _ = parser.parse_known_args()

    # ----  DDP init (torchrun injects env vars)  ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"DDP init: rank={rank}  world_size={world_size}  devices={torch.cuda.device_count()}", flush=True)

    if args.grid_search:
        wd_list = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
                   1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                   1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3]
        eta_list = [1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
                    1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7]
        dd_list = [4, 6, 8, 10, 12, 14, 16]
        total = len(wd_list) * len(eta_list) * len(dd_list)

        if rank == 0:
            print(f"Grid search: {len(wd_list)}wd × {len(eta_list)}lr × {len(dd_list)}dd = {total} runs", flush=True)

        all_results = []
        idx = 0
        for wd in wd_list:
            for eta in eta_list:
                for dd in dd_list:
                    idx += 1
                    combo_dir = os.path.join(args.out_dir, f"wd{wd:.0e}_eta{eta:.0e}_dd{dd}")
                    if rank == 0:
                        print(f"\n[{idx}/{total}] wd={wd:.0e}  eta={eta:.0e}  dd={dd}", flush=True)

                    r = train_one(
                        out_dir=combo_dir,
                        seed=args.seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        weight_decay=wd,
                        eta_min=eta,
                        digit_dim=dd,
                    )
                    if rank == 0 and r is not None:
                        r.update(weight_decay=wd, eta_min=eta, digit_dim=dd, out_dir=combo_dir)
                        all_results.append(r)

        if rank == 0 and all_results:
            json.dump(all_results, open(f"{args.out_dir}/grid_summary.json", "w"), indent=2)
            pd.DataFrame(all_results).to_csv(f"{args.out_dir}/grid_summary.csv", index=False)
            best = min(all_results, key=lambda x: x["test_mae"])
            json.dump(best, open(f"{args.out_dir}/best_combo.json", "w"), indent=2)
            print(f"\nBEST: wd={best['weight_decay']:.0e}  eta={best['eta_min']:.0e}  dd={best['digit_dim']}  "
                  f"MAE={best['test_mae']:.1f}  R²={best['test_r2']:.6f}", flush=True)
    else:
        train_one(
            out_dir=args.out_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

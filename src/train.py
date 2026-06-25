from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from common import (
    Timer,
    append_csv,
    atomic_torch_save,
    ensure_dirs,
    infer_volume_25d,
    load_cached_case,
    load_config,
    masked_metrics,
    make_or_load_split,
    project_path,
    restore_rng_state,
    rng_state,
    set_seed,
)
from dataset import MRIDenoiseTrainDataset
from losses import DenoiseLoss
from model import ResidualUNet2D


def plot_training_curves(log_path: Path, out_path: Path) -> None:
    try:
        import csv
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping curve plot: {exc}")
        return

    if not log_path.exists():
        return
    rows = []
    with log_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_psnr = [float(r["val_psnr"]) for r in rows]
    val_ssim = [float(r["val_ssim"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.5)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_psnr, marker="o", linewidth=1.5)
    axes[1].set_title("Validation PSNR")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, val_ssim, marker="o", linewidth=1.5)
    axes[2].set_title("Validation SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--resume", default="auto", help="'auto' loads ckpt/last.pt, 'none' starts fresh, or a checkpoint path.")
    return parser.parse_args()


@torch.no_grad()
def validate(model: torch.nn.Module, case_ids: list[str], cache_dir: Path, device: torch.device, batch_slices: int, amp: bool) -> tuple[float, float]:
    psnrs, ssims = [], []
    for caseid in tqdm(case_ids, desc="val", leave=False):
        c = load_cached_case(cache_dir, caseid)
        pred = infer_volume_25d(model, c["noisy"].astype(np.float32), device, batch_slices=batch_slices, amp=amp)
        metrics = masked_metrics(pred, c["clean"].astype(np.float32), c["mask"])
        psnrs.append(metrics["psnr"])
        ssims.append(metrics["ssim"])
    return float(np.mean(psnrs)), float(np.mean(ssims))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_psnr: float,
    best_epoch: int,
    no_improve_count: int,
    cfg: dict,
    step_in_epoch: int = 0,
    epoch_complete: bool = True,
    phase: str = "epoch_end",
    train_loss_sum: float = 0.0,
    train_loss_count: int = 0,
) -> None:
    atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "epoch_complete": epoch_complete,
            "phase": phase,
            "train_loss_sum": train_loss_sum,
            "train_loss_count": train_loss_count,
            "best_psnr": best_psnr,
            "best_epoch": best_epoch,
            "no_improve_count": no_improve_count,
            "rng_states": rng_state(),
            "config": cfg,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(int(cfg["seed"]))

    split = make_or_load_split(cfg)
    cache_dir = project_path(cfg["cache_dir"])
    ckpt_dir = project_path(cfg["ckpt_dir"])
    outputs_dir = project_path(cfg["outputs_dir"])
    train_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available; training will be slow.")

    model = ResidualUNet2D(in_ch=3, out_ch=1, base=int(train_cfg["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = int(args.max_epochs or train_cfg["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = DenoiseLoss(
        ssim_weight=float(train_cfg["ssim_weight"]),
        grad_weight=float(train_cfg["grad_loss_weight"]),
        use_gradient=bool(train_cfg["use_gradient_loss"]),
    )

    start_epoch = 1
    resume_step_in_epoch = 0
    resume_phase = "epoch_end"
    resume_loss_sum = 0.0
    resume_loss_count = 0
    global_step = 0
    best_psnr = -float("inf")
    best_epoch = 0
    no_improve_count = 0

    resume_path: Path | None = None
    if args.resume != "none":
        candidate = ckpt_dir / "last.pt" if args.resume == "auto" else project_path(args.resume)
        if candidate.exists():
            resume_path = candidate
    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["amp_scaler"])
        ckpt_epoch = int(ckpt["epoch"])
        epoch_complete = bool(ckpt.get("epoch_complete", True))
        resume_phase = str(ckpt.get("phase", "epoch_end"))
        resume_step_in_epoch = 0 if epoch_complete else int(ckpt.get("step_in_epoch", 0))
        start_epoch = ckpt_epoch + 1 if epoch_complete else ckpt_epoch
        resume_loss_sum = float(ckpt.get("train_loss_sum", 0.0))
        resume_loss_count = int(ckpt.get("train_loss_count", 0))
        global_step = int(ckpt["global_step"])
        best_psnr = float(ckpt["best_psnr"])
        best_epoch = int(ckpt["best_epoch"])
        no_improve_count = int(ckpt["no_improve_count"])
        restore_rng_state(ckpt["rng_states"])
    elif args.resume == "auto":
        print("No ckpt/last.pt found; starting a fresh run.")
    else:
        print("Starting a fresh run.")

    train_ds = MRIDenoiseTrainDataset(
        split["train"],
        cache_dir,
        patch_size=int(train_cfg["patch_size"]),
        samples_per_epoch=int(train_cfg["samples_per_epoch"]),
        lru_cases=int(train_cfg.get("lru_cases_per_worker", 8)),
    )
    num_workers = int(train_cfg["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    batch_size = int(train_cfg["batch_size"])

    def loader_from_step(step_in_epoch: int) -> tuple[DataLoader, int]:
        if step_in_epoch <= 0:
            return train_loader, 0
        start_sample = min(step_in_epoch * batch_size, len(train_ds))
        remaining_ds = Subset(train_ds, range(start_sample, len(train_ds)))
        return (
            DataLoader(
                remaining_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                drop_last=True,
                worker_init_fn=seed_worker,
            ),
            step_in_epoch,
        )

    log_path = outputs_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_psnr", "val_ssim", "lr", "elapsed_hours"]
    timer = Timer()
    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"
    patience = int(train_cfg["early_stop_patience"])
    save_every_steps = int(train_cfg.get("save_every_steps", 100))
    save_every_epochs = int(train_cfg.get("save_every_epochs", 1))

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            current_phase = "train"
            current_step_in_epoch = 0
            loss_sum = resume_loss_sum if epoch == start_epoch and resume_phase == "train" else 0.0
            loss_count = resume_loss_count if epoch == start_epoch and resume_phase == "train" else 0
            recent_losses: list[float] = []

            if not (epoch == start_epoch and resume_phase == "validating"):
                model.train()
                active_loader, step_base = loader_from_step(
                    resume_step_in_epoch if epoch == start_epoch and resume_phase == "train" else 0
                )
                pbar = tqdm(active_loader, desc=f"epoch {epoch}/{max_epochs}")
                for batch_idx, (x, y) in enumerate(pbar):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        pred = model(x)
                        loss = criterion(pred, y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    global_step += 1
                    current_step_in_epoch = step_base + batch_idx + 1
                    loss_value = float(loss.detach().cpu())
                    loss_sum += loss_value
                    loss_count += 1
                    recent_losses.append(loss_value)
                    if save_every_steps > 0 and global_step % save_every_steps == 0:
                        save_checkpoint(
                            last_path,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            global_step,
                            best_psnr,
                            best_epoch,
                            no_improve_count,
                            cfg,
                            step_in_epoch=current_step_in_epoch,
                            epoch_complete=False,
                            phase="train",
                            train_loss_sum=loss_sum,
                            train_loss_count=loss_count,
                        )
                    pbar.set_postfix(
                        loss=f"{np.mean(recent_losses[-50:]):.4f}",
                        saved_step=global_step if save_every_steps > 0 and global_step % save_every_steps == 0 else "",
                    )

                resume_step_in_epoch = 0
                scheduler.step()
                train_loss = float(loss_sum / max(loss_count, 1))
                current_phase = "validating"
            else:
                train_loss = float(resume_loss_sum / max(resume_loss_count, 1))
                current_phase = "validating"

            val_psnr, val_ssim = validate(
                model,
                split["val"],
                cache_dir,
                device,
                batch_slices=int(train_cfg["val_batch_slices"]),
                amp=amp_enabled,
            )

            improved = val_psnr > best_psnr
            if improved:
                best_psnr = val_psnr
                best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1

            early_stop_reached = no_improve_count >= patience
            time_limit_reached = args.max_hours is not None and timer.hours >= float(args.max_hours)
            should_save_last = (
                save_every_epochs <= 1
                or not last_path.exists()
                or epoch % save_every_epochs == 0
                or epoch == max_epochs
                or early_stop_reached
                or time_limit_reached
            )
            if should_save_last:
                save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_psnr,
                    best_epoch,
                    no_improve_count,
                    cfg,
                    step_in_epoch=0,
                    epoch_complete=True,
                    phase="epoch_end",
                    train_loss_sum=0.0,
                    train_loss_count=0,
                )
            if improved:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_psnr,
                    best_epoch,
                    no_improve_count,
                    cfg,
                    step_in_epoch=0,
                    epoch_complete=True,
                    phase="epoch_end",
                    train_loss_sum=0.0,
                    train_loss_count=0,
                )

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_psnr": f"{val_psnr:.6f}",
                "val_ssim": f"{val_ssim:.6f}",
                "lr": f"{scheduler.get_last_lr()[0]:.8g}",
                "elapsed_hours": f"{timer.hours:.4f}",
            }
            append_csv(log_path, row, log_fields)
            plot_training_curves(log_path, outputs_dir / "training_curves.png")
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} val_psnr={val_psnr:.3f} "
                f"val_ssim={val_ssim:.4f} best={best_psnr:.3f}@{best_epoch} no_improve={no_improve_count}"
            )

            if early_stop_reached:
                print(f"Early stopping at epoch {epoch}.")
                break
            if time_limit_reached:
                print(f"Reached --max-hours {args.max_hours}; last checkpoint is saved.")
                break
            resume_phase = "epoch_end"
            resume_loss_sum = 0.0
            resume_loss_count = 0
    except KeyboardInterrupt:
        print("\nInterrupted. Saving ckpt/last.pt before exit...", file=sys.stderr)
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_psnr,
            best_epoch,
            no_improve_count,
            cfg,
            step_in_epoch=locals().get("current_step_in_epoch", 0),
            epoch_complete=False,
            phase=locals().get("current_phase", "train"),
            train_loss_sum=locals().get("loss_sum", 0.0),
            train_loss_count=locals().get("loss_count", 0),
        )
        raise


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()

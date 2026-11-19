"""
Author     : Cao Junjia

Training entry point — works for TFT, LSTM, and Informer wrapper.

Usage
-----
python -m src.experiment.train --config configs/tft_etth1_96_24.yaml
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import ETTDatasetManager
from src.model.factory import build_model
from src.utils.io import load_config, save_config
from src.utils.logger import get_logger
from src.utils.seed import fix_seed


# ─────────────────────────── helpers ─────────────────────────────────────────

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _forward(model, batch, device: str, model_name: str):
    """Unified forward pass — handles different model signatures."""
    enc_x, enc_t, dec_x, dec_t, target = [b.to(device) for b in batch]

    if model_name == "lstm":
        pred = model(enc_x)                            # LSTM ignores enc_t/dec
    else:
        pred = model(enc_x, enc_t, dec_x, dec_t)      # TFT / Informer

    return pred, target


# ─────────────────────────── main training loop ───────────────────────────────

def train(cfg: dict) -> str:
    fix_seed(cfg.get("seed", 42))
    device = _device()

    logger = get_logger(
        f"{cfg['model']}_{cfg['dataset']}_{cfg['seq_len']}_{cfg['pred_len']}",
        log_dir="results/logs",
    )
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    # ── data ─────────────────────────────────────────────────────────────────
    mgr = ETTDatasetManager(
        csv_path  = cfg["data_path"],
        seq_len   = cfg["seq_len"],
        label_len = cfg["label_len"],
        pred_len  = cfg["pred_len"],
        freq      = cfg.get("freq", "h"),
        target    = cfg.get("target", "OT"),
    )
    train_loader = mgr.get_loader("train", batch_size=cfg.get("batch_size", 32))
    val_loader   = mgr.get_loader("val",   batch_size=cfg.get("batch_size", 32), shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_name       = cfg["model"],
        n_features       = mgr.n_features,
        n_time_features  = mgr.n_time_features,
        seq_len          = cfg["seq_len"],
        label_len        = cfg["label_len"],
        pred_len         = cfg["pred_len"],
        freq             = cfg.get("freq", "h"),
        d_model          = cfg.get("d_model", 128),
        n_heads          = cfg.get("n_heads", 4),
        lstm_layers      = cfg.get("lstm_layers", 2),
        hidden_size      = cfg.get("hidden_size", 256),
        dropout          = cfg.get("dropout", 0.1),
    ).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-5
    )
    scheduler = ReduceLROnPlateau(
        optimiser, mode="min", patience=2, factor=0.5
    )
    criterion = nn.MSELoss()

    # ── checkpoint path ───────────────────────────────────────────────────────
    ckpt_dir = os.path.join(
        "results/checkpoints",
        f"{cfg['model']}_{cfg['dataset']}_{cfg['seq_len']}_{cfg['pred_len']}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    save_config(cfg, os.path.join(ckpt_dir, "config.yaml"))

    # ── training loop ─────────────────────────────────────────────────────────
    n_epochs     = cfg.get("epochs", 30)
    patience     = cfg.get("patience", 5)
    best_val     = float("inf")
    patience_cnt = 0

    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            pred, target = _forward(model, batch, device, cfg["model"])
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pred, target = _forward(model, batch, device, cfg["model"])
                val_loss += criterion(pred, target).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch:03d}/{n_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  ✓ Best model saved (val={best_val:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    # Save loss history
    import json
    with open(os.path.join(ckpt_dir, "loss_history.json"), "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f, indent=2)

    logger.info(f"Training complete. Best checkpoint: {best_ckpt}")
    return best_ckpt


# ─────────────────────────── CLI ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g. configs/tft_etth1_96_24.yaml)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

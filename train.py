import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.gap_net import Sentinel2ResUNet
from datasets.raster_datasets import S2S1DSMTileFolderDataset
import config

def sse_and_count(pred: torch.Tensor, target: torch.Tensor):
    # pred/target: [B,1,H,W]
    diff = pred - target
    sse = torch.sum(diff * diff).item()
    n = diff.numel()
    return sse, n


# Logging setup

log_path = getattr(config, "LOG_PATH", "logs/train_tiled.log")
Path(log_path).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
logging.info("Starting Sen2Height tiled training run")


# Device + threads

torch.set_num_threads(getattr(config, "NUM_THREADS", 30))
device = torch.device(getattr(config, "DEVICE", "cuda"))  # "cuda" if available
logging.info(f"Using device: {device}")


# -------------------------
# Datasets
# -------------------------
# Expected folder structure:
# TRAIN_ROOT/
#   S1/*.tif
#   S2/*.tif
#   BDOM/*.tif
# VAL_ROOT/
#   S1/*.tif
#   S2/*.tif
#   BDOM/*.tif
train_ds = S2S1DSMTileFolderDataset(config.TRAIN_ROOT)
val_ds   = S2S1DSMTileFolderDataset(config.VAL_ROOT)


logging.info(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")



# DataLoaders

num_workers = getattr(config, "NUM_WORKERS", 4)

train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)


# Model / loss / optimizer

model = Sentinel2ResUNet(in_channels=config.NUM_BANDS, s1_in_channels=config.S1_BANDS).to(device)
criterion = torch.nn.SmoothL1Loss(beta=0.05).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
use_amp = (device.type=="cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
accum_steps = 4

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = "min",
    factor = 0.5,
    patience = 5,
    threshold = 1e-4,
    min_lr = 1e-6, 
)

best_val_rmse = float("inf")
model_out = getattr(config, "MODEL_OUT", "models/output/model_best.pth")
Path(model_out).parent.mkdir(parents=True, exist_ok=True)


# Training Loop

for epoch in range(config.EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_sse = 0.0
    train_n = 0
    train_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        s2 = batch["s2"].to(device, non_blocking=True)
        s1 = batch["s1"].to(device, non_blocking=True)
        y  = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(s2, s1)
            raw_loss = criterion(pred, y)
            loss = raw_loss / accum_steps

        scaler.scale(loss).backward()

        train_loss_sum += raw_loss.item()
        pred_rmse = pred.detach().float().clamp(0,1)
        diff = pred_rmse - y.float()
        train_sse += torch.sum(diff * diff).item()
        train_n += diff.numel()
        train_batches += 1

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # flush leftover gradients
    if (step + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = train_loss_sum / max(train_batches, 1)
    avg_train_rmse = (train_sse / max(train_n, 1)) ** 0.5


    # Validation

    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    val_sse = 0.0
    val_n = 0

    with torch.no_grad():
        for batch in val_loader:
            s2 = batch["s2"].to(device, non_blocking=True)
            s1 = batch["s1"].to(device, non_blocking=True)
            y  = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(s2, s1)
                vloss = criterion(pred, y)

            val_loss_sum += vloss.item()
            val_batches += 1

            pred_rmse = pred.float().clamp(0, 1)
            batch_sse, batch_n = sse_and_count(pred_rmse, y.float())
            val_sse += batch_sse
            val_n += batch_n

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    avg_val_rmse = (val_sse / max(val_n, 1)) ** 0.5

    scheduler.step(avg_val_rmse)
    current_lr = optimizer.param_groups[0]["lr"]

    logging.info(
        f"Epoch {epoch + 1}/{config.EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if avg_val_rmse < best_val_rmse:
        best_val_rmse = avg_val_rmse
        torch.save(model.state_dict(), model_out)
        logging.info(f"Saved new best model to {model_out}")
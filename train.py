import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from .config import Config
from .dataset import RMOSNPZDataset
from .models import RMOSUNet
from MinkowskiEngine.utils import SparseCollation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default=Config.RMOS_ROOT,
        help="Root of preprocessed RMOS data (with npz/, train.txt, val.txt)",
    )
    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    p.add_argument("--max_iter", type=int, default=Config.MAX_ITER)
    p.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    p.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    p.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--log_dir",
        type=str,
        default="./runs/rmos",
        help="Where to save checkpoints and logs",
    )
    p.add_argument(
        "--val_freq",
        type=int,
        default=Config.VAL_FREQ,
        help="Validate every N iterations",
    )
    p.add_argument(
        "--print_freq",
        type=int,
        default=Config.PRINT_FREQ,
        help="Print training loss every N iterations",
    )
    p.add_argument(
        "--save_freq",
        type=int,
        default=5000,
        help="Save last checkpoint every N iterations",
    )
    return p.parse_args()


def make_loaders(args):
    train_list = os.path.join(args.data_root, "train.txt")
    val_list = os.path.join(args.data_root, "val.txt")

    train_ds = RMOSNPZDataset(args.data_root, train_list)
    val_ds = RMOSNPZDataset(args.data_root, val_list)

    collate_fn = SparseCollation(limit_numpoints=-1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def sparse_tensor_from_batch(batch, device):
    coords, feats, labels = batch
    coords = coords.to(device)
    feats = feats.to(device)
    labels = labels.to(device).long()
    x = ME.SparseTensor(feats, coordinates=coords, device=device)
    return x, labels


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()

    total_correct = 0
    total_seen = 0

    # confusion matrix: rows = gt, cols = pred
    hist = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        x, labels = sparse_tensor_from_batch(batch, device)
        logits = model(x).F  # (N, C)
        preds = logits.argmax(1)

        mask = labels != -1
        if mask.sum() == 0:
            continue

        gt = labels[mask].view(-1)
        pr = preds[mask].view(-1)

        total_correct += (gt == pr).sum().item()
        total_seen += gt.numel()

        # update confusion matrix
        for c in range(num_classes):
            for c2 in range(num_classes):
                hist[c, c2] += ((gt == c) & (pr == c2)).sum().item()

    acc = total_correct / max(total_seen, 1)

    # per-class IoU
    ious = []
    for c in range(num_classes):
        tp = hist[c, c].item()
        fn = (hist[c, :].sum() - tp).item()
        fp = (hist[:, c].sum() - tp).item()
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
        else:
            ious.append(float("nan"))

    # mIoU over valid classes
    valid_ious = [iou for iou in ious if not (iou != iou)]  # filter NaNs
    miou = sum(valid_ious) / max(len(valid_ious), 1)

    # assume class 1 = moving (adjust if different in your config)
    moving_class = 1 if num_classes > 1 else 0
    tp = hist[moving_class, moving_class].item()
    fp = (hist[:, moving_class].sum() - tp).item()
    fn = (hist[moving_class, :].sum() - tp).item()

    iou_moving = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    prec_moving = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_moving = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec_moving + rec_moving > 0:
        f1_moving = 2 * prec_moving * rec_moving / (prec_moving + rec_moving)
    else:
        f1_moving = 0.0

    print(
        f"[VAL] Acc: {acc:.4f} | mIoU: {miou:.4f} | "
        f"IoU(moving): {iou_moving:.4f} | "
        f"Prec(moving): {prec_moving:.4f} | "
        f"Rec(moving): {rec_moving:.4f} | "
        f"F1(moving): {f1_moving:.4f}"
    )

    model.train()

    return {
        "acc": acc,
        "miou": miou,
        "iou_moving": iou_moving,
        "prec_moving": prec_moving,
        "rec_moving": rec_moving,
        "f1_moving": f1_moving,
    }


def save_checkpoint(path, model, optimizer, global_step, best_iou_moving):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_iou_moving": best_iou_moving,
        },
        path,
    )


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(args)

    model = RMOSUNet(
        in_channels=1,
        out_channels=args.num_classes,
        D=3,
        base_channels=Config.BASE_CHANNELS,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Simple LR schedule: decay every 10k iters by 0.5 (tune as you like)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=0.5
    )

    best_iou_moving = 0.0
    global_step = 0

    model.train()

    while global_step < args.max_iter:
        for batch in train_loader:
            if global_step >= args.max_iter:
                break

            x, labels = sparse_tensor_from_batch(batch, device)

            logits = model(x)  # SparseTensor: (N, C)
            preds = logits.F

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % args.print_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[Iter {global_step}] loss={loss.item():.4f} lr={lr:.6f}")

            if global_step > 0 and global_step % args.val_freq == 0:
                metrics = evaluate(
                    model, val_loader, device, args.num_classes
                )

                # Track best model by IoU(moving)
                if metrics["iou_moving"] > best_iou_moving:
                    best_iou_moving = metrics["iou_moving"]
                    best_path = os.path.join(args.log_dir, "best_moving.pth")
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        global_step,
                        best_iou_moving,
                    )
                    print(
                        f"  [BEST] Updated best IoU(moving): "
                        f"{best_iou_moving:.4f} at iter {global_step}"
                    )

            if global_step > 0 and global_step % args.save_freq == 0:
                last_path = os.path.join(args.log_dir, "last.pth")
                save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    global_step,
                    best_iou_moving,
                )
                print(f"  [CKPT] Saved last checkpoint at iter {global_step}")

            global_step += 1

    print("Training completed.")
    # Final eval
    _ = evaluate(model, val_loader, device, args.num_classes)


if __name__ == "__main__":
    main()

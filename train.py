# rmos/train.py
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import MinkowskiEngine as ME
from MinkowskiEngine.utils import SparseCollation

from .config import Config
from .dataset import RMOSNPZDataset, RMOSSequenceNPZDataset
from .models import RMOSUNet, RMOS4DUNetV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=Config.RMOS_ROOT)
    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=Config.NUM_WORKERS)
    p.add_argument("--max_iter", type=int, default=Config.MAX_ITER)
    p.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    p.add_argument("--weight_decay", type=float, default=Config.WEIGHT_DECAY)
    p.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_dir", type=str, default="./runs/rmos")
    p.add_argument("--val_freq", type=int, default=Config.VAL_FREQ)
    p.add_argument("--print_freq", type=int, default=Config.PRINT_FREQ)
    p.add_argument("--save_freq", type=int, default=5000)

    # temporal stacking
    p.add_argument("--n_frames", type=int, default=1,
                   help=">1 enables 4D spatio-temporal training")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="stride for previous frames in sequence")

    # model width
    p.add_argument("--base_channels", type=int, default=Config.BASE_CHANNELS)

    # collation / point limits
    p.add_argument("--limit_numpoints", type=int, default=-1,
                   help="Max total points per batch before SparseCollation truncates. "
                        "-1 = no limit (but may OOM).")
    p.add_argument("--max_points_per_sample", type=int, default=0,
                   help="Optional hard cap per sample AFTER collation. "
                        "0 = no cap. Useful to keep memory stable in 4D.")

    # NEW: temporal feature on input
    p.add_argument("--time_feat", type=str, default="none",
                   choices=["none", "scalar"],
                   help="How to encode time as an extra feature channel for 4D.")

    # NEW: loss shaping
    p.add_argument("--use_balanced_loss", action="store_true",
                   help="Use class-balanced CE + moving-class Dice loss.")
    p.add_argument("--moving_weight", type=float, default=2.0,
                   help="Class weight for the moving class in CE.")
    p.add_argument("--dice_weight", type=float, default=1.0,
                   help="Lambda for moving-class Dice loss.")

    return p.parse_args()


def make_loaders(args):
    train_list = os.path.join(args.data_root, "train.txt")
    val_list   = os.path.join(args.data_root, "val.txt")

    if args.n_frames > 1:
        train_ds = RMOSSequenceNPZDataset(
            args.data_root, train_list,
            n_frames=args.n_frames,
            frame_stride=args.frame_stride,
            time_feat=args.time_feat,
        )
        val_ds   = RMOSSequenceNPZDataset(
            args.data_root, val_list,
            n_frames=args.n_frames,
            frame_stride=args.frame_stride,
            time_feat=args.time_feat,
        )
    else:
        train_ds = RMOSNPZDataset(args.data_root, train_list)
        val_ds   = RMOSNPZDataset(args.data_root, val_list)

    # If you truly want "no truncation", give a huge limit.
    limit = args.limit_numpoints
    if limit is None or limit < 0:
        limit = 10**9

    collate_fn = SparseCollation(limit_numpoints=limit)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False
    )

    print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)
    print(f"[INFO] Train iters/epoch (approx): {len(train_loader)} | Val iters: {len(val_loader)}", flush=True)
    return train_loader, val_loader


def maybe_subsample(coords, feats, labels, max_points_per_sample: int):
    """Randomly subsample to max_points_per_sample (keeps alignment)."""
    if max_points_per_sample is None or max_points_per_sample <= 0:
        return coords, feats, labels
    n = coords.shape[0]
    if n <= max_points_per_sample:
        return coords, feats, labels

    idx = torch.randperm(n, device=coords.device)[:max_points_per_sample]
    return coords[idx], feats[idx], labels[idx]


def sparse_tensor_from_batch(batch, device, max_points_per_sample=0):
    coords, feats, labels = batch
    coords = coords.to(device)
    feats  = feats.to(device)
    labels = labels.to(device).long()

    coords, feats, labels = maybe_subsample(
        coords, feats, labels, max_points_per_sample
    )

    x = ME.SparseTensor(feats, coordinates=coords, device=device)
    return x, labels


@torch.no_grad()
def evaluate(model, loader, device, num_classes, max_points_per_sample=0):
    model.eval()
    total_correct, total_seen = 0, 0
    hist = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        x, labels = sparse_tensor_from_batch(
            batch, device, max_points_per_sample=max_points_per_sample
        )
        logits = model(x).F
        preds = logits.argmax(1)

        mask = labels != -1
        if mask.sum() == 0:
            continue

        gt = labels[mask].view(-1)
        pr = preds[mask].view(-1)

        total_correct += (gt == pr).sum().item()
        total_seen += gt.numel()

        for c in range(num_classes):
            for c2 in range(num_classes):
                hist[c, c2] += ((gt == c) & (pr == c2)).sum().item()

    acc = total_correct / max(total_seen, 1)

    ious = []
    for c in range(num_classes):
        tp = hist[c, c].item()
        fn = (hist[c, :].sum() - tp).item()
        fp = (hist[:, c].sum() - tp).item()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))

    valid_ious = [iou for iou in ious if not (iou != iou)]
    miou = sum(valid_ious) / max(len(valid_ious), 1)

    moving_class = 1 if num_classes > 1 else 0
    tp = hist[moving_class, moving_class].item()
    fp = (hist[:, moving_class].sum() - tp).item()
    fn = (hist[moving_class, :].sum() - tp).item()

    iou_moving = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    prec_moving = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_moving  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_moving   = (2 * prec_moving * rec_moving / (prec_moving + rec_moving)
                   if (prec_moving + rec_moving) > 0 else 0.0)

    print(
        f"[VAL] Acc: {acc:.4f} | mIoU: {miou:.4f} | "
        f"IoU(moving): {iou_moving:.4f} | "
        f"Prec(moving): {prec_moving:.4f} | "
        f"Rec(moving): {rec_moving:.4f} | "
        f"F1(moving): {f1_moving:.4f}",
        flush=True
    )
    model.train()
    return dict(acc=acc, miou=miou, iou_moving=iou_moving,
                prec_moving=prec_moving, rec_moving=rec_moving, f1_moving=f1_moving)


def save_checkpoint(path, model, optimizer, global_step, best_iou_moving, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_iou_moving": best_iou_moving,
    }
    if meta:
        ckpt["meta"] = meta
    torch.save(ckpt, path)


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # For 4D, batch_size>1 almost always collapses to 1 anyway.
    if args.n_frames > 1 and args.batch_size > 1:
        print(f"[WARN] n_frames={args.n_frames} makes samples large. "
              f"Setting batch_size=1 to avoid collation truncation/noise.",
              flush=True)
        args.batch_size = 1

    train_loader, val_loader = make_loaders(args)

    # Decide input feature dimension
    if args.n_frames > 1:
        in_channels = 1 if args.time_feat == "none" else 2
        model = RMOS4DUNetV2(
            in_channels=in_channels,
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            D=4
        ).to(device)
        print(f"[INFO] Training 4D model (V2) with n_frames={args.n_frames} "
              f"and in_channels={in_channels} (time_feat={args.time_feat})",
              flush=True)
    else:
        # 3D baseline uses only spatial frame, 1-channel features
        model = RMOSUNet(
            in_channels=1,
            out_channels=args.num_classes,
            D=3,
            base_channels=args.base_channels
        ).to(device)
        print("[INFO] Training 3D model", flush=True)

    # Loss: either plain CE or balanced CE + moving Dice
    if args.use_balanced_loss:
        class_weights = torch.ones(args.num_classes, device=device)
        if args.num_classes > 1:
            # assume class 1 is "moving"
            class_weights[1] = args.moving_weight
        criterion_ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-1,
        )
        print(f"[INFO] Using balanced loss: CE(weighted, moving_weight={args.moving_weight}) "
              f"+ dice_weight={args.dice_weight}", flush=True)
    else:
        criterion_ce = nn.CrossEntropyLoss(ignore_index=-1)
        print("[INFO] Using standard CrossEntropyLoss", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    best_iou_moving = 0.0

    print("[INFO] Entering training loop.", flush=True)
    model.train()
    while global_step < args.max_iter:
        for bi, batch in enumerate(train_loader):
            t_data0 = time.time()
            global_step += 1

            x, labels = sparse_tensor_from_batch(
                batch, device, max_points_per_sample=args.max_points_per_sample
            )
            t_data = time.time() - t_data0

            # -------- forward + loss --------
            t_fwd0 = time.time()
            out = model(x)
            logits = out.F

            if not args.use_balanced_loss:
                loss = criterion_ce(logits, labels)
            else:
                # mask out ignored labels
                mask = labels != -1
                if mask.any():
                    logits_valid = logits[mask]
                    labels_valid = labels[mask]

                    ce_loss = criterion_ce(logits_valid, labels_valid)

                    # moving-class Dice (class 1)
                    if args.num_classes > 1:
                        probs = torch.softmax(logits_valid, dim=1)
                        p_m = probs[:, 1]
                        g_m = (labels_valid == 1).float()
                        eps = 1e-5
                        inter = (p_m * g_m).sum()
                        union = p_m.sum() + g_m.sum()
                        dice = (2 * inter + eps) / (union + eps)
                        dice_loss = 1.0 - dice
                    else:
                        dice_loss = torch.tensor(0.0, device=device)

                    loss = ce_loss + args.dice_weight * dice_loss
                else:
                    # rare case if everything is ignored
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            t_fwd = time.time() - t_fwd0

            # -------- backward --------
            t_bwd0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            t_bwd = time.time() - t_bwd0

            if global_step <= 3:
                print(f"[DEBUG] First batches. Npoints={x.F.shape[0]} "
                      f"coords_dim={x.C.shape[1]} feat_dim={x.F.shape[1]}",
                      flush=True)

            if global_step % args.print_freq == 0 or global_step <= 3:
                print(f"[STEP {global_step}] loss={loss.item():.4f} | "
                      f"data={t_data:.3f}s fwd={t_fwd:.3f}s bwd+opt={t_bwd:.3f}s",
                      flush=True)

            if global_step % args.val_freq == 0:
                stats = evaluate(
                    model, val_loader, device, args.num_classes,
                    max_points_per_sample=args.max_points_per_sample
                )
                if stats["iou_moving"] > best_iou_moving:
                    best_iou_moving = stats["iou_moving"]
                    save_checkpoint(
                        os.path.join(args.log_dir, "best_moving.pth"),
                        model, optimizer, global_step, best_iou_moving,
                        meta={
                            "n_frames": args.n_frames,
                            "frame_stride": args.frame_stride,
                            "max_points_per_sample": args.max_points_per_sample,
                            "time_feat": args.time_feat,
                            "use_balanced_loss": args.use_balanced_loss,
                            "moving_weight": args.moving_weight,
                            "dice_weight": args.dice_weight,
                        }
                    )
                    print(f"[INFO] New best iou_moving={best_iou_moving:.4f}", flush=True)

            if global_step % args.save_freq == 0:
                save_checkpoint(
                    os.path.join(args.log_dir, "last.pth"),
                    model, optimizer, global_step, best_iou_moving,
                    meta={
                        "n_frames": args.n_frames,
                        "frame_stride": args.frame_stride,
                        "max_points_per_sample": args.max_points_per_sample,
                        "time_feat": args.time_feat,
                        "use_balanced_loss": args.use_balanced_loss,
                        "moving_weight": args.moving_weight,
                        "dice_weight": args.dice_weight,
                    }
                )

            if global_step >= args.max_iter:
                break


if __name__ == "__main__":
    main()

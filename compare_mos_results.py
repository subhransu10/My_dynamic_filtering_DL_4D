#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# ---- Use your dataset to ensure GT aligns with model input (3D or 4D) ----
from rmos.dataset import RMOSNPZDataset, RMOSSequenceNPZDataset


# ---------- Metrics ----------
def confusion_binary(gt: np.ndarray, pred: np.ndarray, pos: int = 1) -> Tuple[int, int, int, int]:
    gt_pos = (gt == pos)
    pr_pos = (pred == pos)
    tp = int(np.sum(gt_pos & pr_pos))
    fp = int(np.sum(~gt_pos & pr_pos))
    fn = int(np.sum(gt_pos & ~pr_pos))
    tn = int(np.sum(~gt_pos & ~pr_pos))
    return tp, fp, fn, tn

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0

def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + fp + fn + tn)
    prec = safe_div(tp, tp + fp)
    rec  = safe_div(tp, tp + fn)
    f1   = safe_div(2 * prec * rec, prec + rec)
    iou_moving = safe_div(tp, tp + fp + fn)

    # binary mIoU = mean(IoU_pos, IoU_neg)
    iou_static = safe_div(tn, tn + fp + fn)
    miou = 0.5 * (iou_moving + iou_static)

    return {
        "acc": acc,
        "miou": miou,
        "iou_moving": iou_moving,
        "prec_moving": prec,
        "rec_moving": rec,
        "f1_moving": f1,
    }

# ---------- Pred loading ----------
def load_pred(pred_path: Path, fmt: str) -> np.ndarray:
    if fmt == "npy":
        return np.load(pred_path).astype(np.int32)

    if fmt == "npz":
        z = np.load(pred_path)
        if "pred" in z:   return z["pred"].astype(np.int32)
        if "labels" in z: return z["labels"].astype(np.int32)
        raise KeyError(f"{pred_path} npz has no 'pred' or 'labels'")

    if fmt == "label_u32":
        arr = np.fromfile(pred_path, dtype=np.uint32)
        return (arr & 0xFFFF).astype(np.int32)

    if fmt == "bin_u8":
        return np.fromfile(pred_path, dtype=np.uint8).astype(np.int32)

    if fmt == "txt":
        return np.loadtxt(pred_path, dtype=np.int32)

    raise ValueError(f"Unknown fmt: {fmt}")

def map_to_binary(pred: np.ndarray, mapping: str, pos_class: int = 1) -> np.ndarray:
    """
    mapping options:
      - already_binary
      - equals:<k>
      - movable_set:<comma_separated_ints>
    """
    if mapping == "already_binary":
        return pred.astype(np.int32)

    if mapping.startswith("equals:"):
        k = int(mapping.split(":", 1)[1])
        out = np.zeros_like(pred, dtype=np.int32)
        out[pred == k] = pos_class
        return out

    if mapping.startswith("movable_set:"):
        ids = mapping.split(":", 1)[1]
        movable = set(int(x) for x in ids.split(",") if x.strip() != "")
        out = np.zeros_like(pred, dtype=np.int32)
        if movable:
            m = np.isin(pred, np.array(sorted(list(movable)), dtype=np.int32))
            out[m] = pos_class
        return out

    raise ValueError(f"Unknown mapping: {mapping}")


def build_dataset(data_root: str, split: str, n_frames: int, frame_stride: int, time_feat: str):
    list_path = os.path.join(data_root, f"{split}.txt")
    if n_frames > 1:
        return RMOSSequenceNPZDataset(
            data_root, list_path,
            n_frames=n_frames,
            frame_stride=frame_stride,
            time_feat=time_feat,
        )
    return RMOSNPZDataset(data_root, list_path)


def main():
    ap = argparse.ArgumentParser("Compare MOS results across models using RMOS datasets (3D or 4D aligned GT).")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])

    # IMPORTANT: must match how you exported predictions (3D vs 4D)
    ap.add_argument("--n_frames", type=int, default=1)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--time_feat", type=str, default="none", choices=["none", "scalar"])

    ap.add_argument("--pos_class", type=int, default=1)
    ap.add_argument("--ignore_label", type=int, default=-1)

    ap.add_argument("--csv_out", type=str, default=None)

    # Repeated: --model name=... pred_root=... fmt=... pattern=... mapping=...
    ap.add_argument("--model", action="append", nargs="+", required=True)

    args = ap.parse_args()

    ds = build_dataset(args.data_root, args.split, args.n_frames, args.frame_stride, args.time_feat)

    # Parse model specs
    models = []
    for spec_parts in args.model:
        spec = {}
        for kv in spec_parts:
            k, v = kv.split("=", 1)
            spec[k] = v
        for k in ["name", "pred_root", "fmt", "pattern", "mapping"]:
            if k not in spec:
                raise ValueError(f"Missing {k} in --model spec: {spec}")
        models.append(spec)

    results = []
    for m in models:
        name = m["name"]
        pred_root = Path(m["pred_root"])
        fmt = m["fmt"]
        pattern = m["pattern"]
        mapping = m["mapping"]

        TP = FP = FN = TN = 0
        missing = 0
        shape_mismatch = 0

        for i in range(len(ds)):
            # ds returns coords, feats, labels (labels includes -1 for past frames in 4D)
            coords, feats, labels = ds[i]
            gt = labels.astype(np.int32)

            keep = (gt != args.ignore_label) if args.ignore_label is not None else np.ones_like(gt, dtype=bool)
            if keep.sum() == 0:
                continue

            # key to locate prediction file:
            # For both 3D and 4D we recommend exporting ONE file per sample keyed by the current-frame rel path.
            rel = ds.files[i]  # the current frame path stored in split list
            pred_path = pred_root / pattern.format(rel=rel, stem=Path(rel).stem, idx=i)

            if not pred_path.exists():
                missing += 1
                continue

            pred_raw = load_pred(pred_path, fmt=fmt)

            if pred_raw.shape[0] != gt.shape[0]:
                shape_mismatch += 1
                continue

            pred_bin = map_to_binary(pred_raw, mapping=mapping, pos_class=args.pos_class)

            tp, fp, fn, tn = confusion_binary(gt[keep], pred_bin[keep], pos=args.pos_class)
            TP += tp; FP += fp; FN += fn; TN += tn

        met = compute_metrics(TP, FP, FN, TN)
        met.update({"name": name, "missing_frames": missing, "shape_mismatch": shape_mismatch})
        results.append(met)

    def pct(x): return f"{x*100:.2f}"

    print(f"\n=== MOS Comparison (split={args.split}, n_frames={args.n_frames}, time_feat={args.time_feat}) ===")
    print("name\tAcc\tmIoU\tIoU(mov)\tPrec(mov)\tRec(mov)\tF1(mov)\tMissing\tShapeMismatch")
    for r in results:
        print(
            f"{r['name']}\t"
            f"{pct(r['acc'])}\t"
            f"{pct(r['miou'])}\t"
            f"{pct(r['iou_moving'])}\t"
            f"{pct(r['prec_moving'])}\t"
            f"{pct(r['rec_moving'])}\t"
            f"{pct(r['f1_moving'])}\t"
            f"{r['missing_frames']}\t"
            f"{r['shape_mismatch']}"
        )

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        header = ["name","acc","miou","iou_moving","prec_moving","rec_moving","f1_moving","missing_frames","shape_mismatch"]
        lines = [",".join(header)]
        for r in results:
            lines.append(",".join(str(r[h]) for h in header))
        out.write_text("\n".join(lines) + "\n")
        print("[INFO] wrote", out)

if __name__ == "__main__":
    main()

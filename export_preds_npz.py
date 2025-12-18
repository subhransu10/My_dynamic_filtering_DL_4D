# rmos/export_preds_npz.py
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import MinkowskiEngine as ME

from rmos.models import RMOSUNet, RMOS4DUNetV2
from rmos.dataset import RMOSNPZDataset, RMOSSequenceNPZDataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--n_frames", type=int, default=1)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--time_feat", default="none", choices=["none", "scalar"])

    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--num_classes", type=int, default=2)

    ap.add_argument("--max_points_per_sample", type=int, default=0)
    return ap.parse_args()

def maybe_subsample(coords, feats, max_points):
    if max_points <= 0 or coords.shape[0] <= max_points:
        return coords, feats, None
    idx = np.random.permutation(coords.shape[0])[:max_points]
    return coords[idx], feats[idx], idx

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    list_path = Path(args.data_root) / f"{args.split}.txt"
    rels = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]

    # dataset just for loading NPZs the same way training does
    if args.n_frames > 1:
        ds = RMOSSequenceNPZDataset(
            args.data_root, str(list_path),
            n_frames=args.n_frames,
            frame_stride=args.frame_stride,
            time_feat=args.time_feat
        )
        model = RMOS4DUNetV2(
            in_channels=1 + (1 if args.time_feat == "scalar" else 0),
            num_classes=args.num_classes,
            base_ch=args.base_channels,
            D=4
        ).to(device)
    else:
        ds = RMOSNPZDataset(args.data_root, str(list_path))
        model = RMOSUNet(
            in_channels=1 + (1 if args.time_feat == "scalar" else 0),
            out_channels=args.num_classes,
            D=3,
            base_channels=args.base_channels
        ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for i, rel in enumerate(rels):
        coords, feats, _labels = ds[i]  # labels not needed for export
        # optional subsample (NOTE: if you subsample here, preds won't align to GT anymore)
        if args.max_points_per_sample > 0:
            coords, feats, _ = maybe_subsample(coords, feats, args.max_points_per_sample)

        coords = torch.from_numpy(coords).int()
        feats = torch.from_numpy(feats).float()

        # add batch index column (ME expects [b,x,y,z,(t)])
        b = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
        coords = torch.cat([b, coords], dim=1).to(device)
        feats = feats.to(device)

        x = ME.SparseTensor(feats, coordinates=coords, device=device)
        logits = model(x).F
        pred = logits.argmax(1).detach().cpu().numpy().astype(np.int32)

        # save to out_dir/npz/08/000000.npy (matches rel without extension)
        rel_noext = str(Path(rel).with_suffix(""))  # npz/08/000000
        out_path = out_root / f"{rel_noext}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, pred)

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(rels)}] wrote {out_path}")

    print("[DONE] Exported predictions to", out_root)

if __name__ == "__main__":
    main()

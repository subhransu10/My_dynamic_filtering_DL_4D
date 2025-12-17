# supports 4D as opposed to the previous 3D only version
import os
import argparse
import numpy as np
import torch
import MinkowskiEngine as ME

from .config import Config
from .dataset import RMOSNPZDataset, RMOSSequenceNPZDataset
from .models import RMOSUNet, RMOS4DUNet


def parse_args():
    p = argparse.ArgumentParser("RMOS Inference & Visualization")
    p.add_argument("--data_root", type=str, default=Config.RMOS_ROOT)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out_ply", type=str, default="vis_sample.ply")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--moving_class", type=int, default=1)

    # NEW
    p.add_argument("--n_frames", type=int, default=1)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--base_channels", type=int, default=Config.BASE_CHANNELS)

    return p.parse_args()


def build_dataset(args):
    list_path = os.path.join(args.data_root, f"{args.split}.txt")
    if args.n_frames > 1:
        return RMOSSequenceNPZDataset(args.data_root, list_path,
                                      n_frames=args.n_frames,
                                      frame_stride=args.frame_stride)
    return RMOSNPZDataset(args.data_root, list_path)


def load_model(args, device):
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model_state", ckpt)
    meta = ckpt.get("meta", {})

    n_frames_ckpt = int(meta.get("n_frames", args.n_frames))

    if n_frames_ckpt > 1:
        model = RMOS4DUNet(
            in_channels=1, out_channels=Config.NUM_CLASSES,
            base_channels=args.base_channels
        ).to(device)
        print(f"[INFO] Loaded 4D model (n_frames={n_frames_ckpt})")
    else:
        model = RMOSUNet(
            in_channels=1, out_channels=Config.NUM_CLASSES,
            D=3, base_channels=args.base_channels
        ).to(device)
        print("[INFO] Loaded 3D model")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = build_dataset(args)
    coords, feats, labels = ds[args.index]

    coords = torch.from_numpy(coords).int()
    feats = torch.from_numpy(feats).float()

    # Add batch index = 0 to coords
    bcol = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    coords_b = torch.cat([bcol, coords], dim=1)

    x = ME.SparseTensor(feats, coordinates=coords_b, device=device)
    model = load_model(args, device)

    out = model(x).F
    pred = out.argmax(1).cpu().numpy()

    print("[INFO] pred shape:", pred.shape, "labels shape:", labels.shape)


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import torch
import MinkowskiEngine as ME

from .config import Config
from .dataset import RMOSNPZDataset
from .models import RMOSUNet


def parse_args():
    p = argparse.ArgumentParser("RMOS Inference & Visualization")

    p.add_argument(
        "--data_root",
        type=str,
        default=Config.RMOS_ROOT,
        help="Root of preprocessed RMOS data (npz/, train.txt, val.txt)",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. runs/rmos/best_moving.pth or last.pth)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to sample from",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the chosen split",
    )
    p.add_argument(
        "--out_ply",
        type=str,
        default="vis_sample.ply",
        help="Output .ply path",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    p.add_argument(
        "--moving_class",
        type=int,
        default=1,
        help="Label id used for 'moving' class in training",
    )

    return p.parse_args()


def build_dataset(data_root: str, split: str) -> RMOSNPZDataset:
    if split == "train":
        list_path = os.path.join(data_root, "train.txt")
    else:  # "val"
        list_path = os.path.join(data_root, "val.txt")

    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Split file not found: {list_path}")

    return RMOSNPZDataset(data_root, list_path)


def to_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if dtype is not None else x
    x = torch.from_numpy(x)
    return x.to(dtype) if dtype is not None else x


def load_model(ckpt_path: str, num_classes: int, device: torch.device) -> RMOSUNet:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure you trained with --log_dir ./runs/rmos "
            f"so best_moving.pth / last.pth exists."
        )

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # support both {"model_state": ...} and raw state_dict
    state_dict = ckpt.get("model_state", ckpt)

    model = RMOSUNet(
        in_channels=1,
        out_channels=num_classes,
        D=3,
        base_channels=Config.BASE_CHANNELS,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict: {unexpected}")

    model.eval()
    return model


def sparse_forward(model, coords, feats, device):
    """
    coords: (N, 4) int tensor [b, x, y, z]
    feats:  (N, C) float tensor
    """
    x = ME.SparseTensor(
        features=feats.to(device),
        coordinates=coords.to(device),
        device=device,
    )
    with torch.no_grad():
        logits = model(x).F  # (N, num_classes)
    preds = logits.argmax(1)
    return preds.cpu().numpy()


def write_ply_xyzrgb(path, xyz, rgb):
    """
    xyz: (N, 3) float
    rgb: (N, 3) uint8
    """
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def build_colors(coords_np, labels, preds, moving_class: int):
    """
    Color scheme (pointwise):
      - red   (255, 0, 0): predicted moving & correct
      - gray  (180,180,180): predicted static & correct
      - green (0,255,0): misclassified (pred != gt)
      - dark gray (80,80,80): label == -1 (ignored)

    If labels are None or size mismatch:
      - red for predicted moving
      - gray for predicted non-moving
    """
    N = coords_np.shape[0]
    rgb = np.zeros((N, 3), dtype=np.uint8)

    # no reliable GT: only encode predictions
    if labels is None or labels.numel() != N:
        mov_mask = preds == moving_class
        rgb[mov_mask] = np.array([255, 0, 0], dtype=np.uint8)
        rgb[~mov_mask] = np.array([180, 180, 180], dtype=np.uint8)
        return rgb

    labels_np = labels.cpu().numpy()
    valid = labels_np != -1

    correct = (preds == labels_np) & valid
    mov_correct = correct & (labels_np == moving_class)
    static_correct = correct & (labels_np != moving_class)
    wrong = (preds != labels_np) & valid

    rgb[static_correct] = np.array([180, 180, 180], dtype=np.uint8)
    rgb[mov_correct] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[wrong] = np.array([0, 255, 0], dtype=np.uint8)
    rgb[~valid] = np.array([80, 80, 80], dtype=np.uint8)

    return rgb


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & sample
    ds = build_dataset(args.data_root, args.split)
    print(f"{args.split} dataset size: {len(ds)} samples")

    if args.index < 0 or args.index >= len(ds):
        raise IndexError(f"Index {args.index} out of range for split {args.split}")

    coords, feats, labels = ds[args.index]
    # coords: (N,3) or (N,4), feats: (N,1), labels: (N,) or None

    # ---- to torch tensors ----
    coords = to_tensor(coords, dtype=torch.int32)
    feats = to_tensor(feats, dtype=torch.float32)
    labels = to_tensor(labels, dtype=torch.long) if labels is not None else None

    # ---- ensure 4D coords: [b, x, y, z] ----
    if coords.dim() != 2:
        raise ValueError(f"coords must be (N,3) or (N,4), got {coords.shape}")

    if coords.size(1) == 3:
        # add batch index 0
        batch_col = torch.zeros((coords.size(0), 1), dtype=torch.int32)
        coords = torch.cat([batch_col, coords], dim=1)
    elif coords.size(1) == 4:
        # assume already [b,x,y,z]
        pass
    else:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    num_classes = Config.NUM_CLASSES

    # Model
    model = load_model(args.ckpt, num_classes, device)

    # Inference
    preds = sparse_forward(model, coords, feats, device)

    # Colors
    coords_np = coords[:, 1:4].cpu().numpy().astype(np.float32)
    colors = build_colors(coords_np, labels, preds, args.moving_class)

    # Save PLY
    out_dir = os.path.dirname(args.out_ply)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_ply_xyzrgb(args.out_ply, coords_np, colors)

    print(f"Saved visualization to: {args.out_ply}")
    print(
        "Legend:\n"
        "  red   = moving (correct)\n"
        "  gray  = static (correct)\n"
        "  green = misclassified\n"
        "  dark gray = ignored / invalid"
    )


if __name__ == "__main__":
    main()

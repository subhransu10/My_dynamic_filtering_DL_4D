# rmos/view_rmos_o3d.py
import os
import argparse
import numpy as np
import torch
import MinkowskiEngine as ME

from .config import Config
from .dataset import RMOSNPZDataset, RMOSSequenceNPZDataset
from .models import RMOSUNet, RMOS4DUNet

if "WSL_DISTRO_NAME" in os.environ:
    os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "1")

try:
    import open3d as o3d
    HAS_O3D = True
except Exception as e:
    print("[WARN] Open3D import failed:", e)
    HAS_O3D = False


def parse_args():
    p = argparse.ArgumentParser("RMOS Open3D Viewer")
    p.add_argument("--data_root", type=str, default=Config.RMOS_ROOT)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val"], default="val")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--voxel_size", type=float, default=Config.VOXEL_SIZE)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--moving_class", type=int, default=1)
    p.add_argument("--export_dir", type=str, default="./vis_seq")

    # NEW
    p.add_argument("--n_frames", type=int, default=1)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--base_channels", type=int, default=Config.BASE_CHANNELS)

    return p.parse_args()


def get_dataset(args):
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
        model = RMOS4DUNet(1, Config.NUM_CLASSES, base_channels=args.base_channels).to(device)
        print(f"[INFO] Loaded 4D model (n_frames={n_frames_ckpt})")
    else:
        model = RMOSUNet(1, Config.NUM_CLASSES, D=3, base_channels=args.base_channels).to(device)
        print("[INFO] Loaded 3D model")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference_for_index(ds, idx, model, device):
    coords, feats, labels = ds[idx]
    coords = torch.from_numpy(coords).int()
    feats  = torch.from_numpy(feats).float()

    bcol = torch.zeros((coords.shape[0], 1), dtype=torch.int32)
    coords_b = torch.cat([bcol, coords], dim=1)

    x = ME.SparseTensor(feats, coordinates=coords_b, device=device)
    logits = model(x).F
    pred = logits.argmax(1).cpu().numpy()

    return coords.cpu().numpy(), labels, pred


def export_ply(path, pts, colors):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pc)


def make_colors(pts, labels, preds, moving_class=1):
    # red=pred moving, green=gt moving, blue=correct static
    colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
    gt_mv = labels == moving_class
    pr_mv = preds == moving_class

    colors[pr_mv] = np.array([1, 0, 0])     # predicted moving
    colors[gt_mv] = np.array([0, 1, 0])     # gt moving
    colors[gt_mv & pr_mv] = np.array([1, 1, 0])  # correct moving
    return colors


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = get_dataset(args)
    model = load_model(args, device)

    start = args.start_index
    end = args.end_index if args.end_index is not None else len(ds)
    os.makedirs(args.export_dir, exist_ok=True)

    for idx in range(start, end):
        pts_voxel, labels, preds = run_inference_for_index(ds, idx, model, device)

        # viewer expects xyz only
        pts_xyz = pts_voxel[:, :3] * args.voxel_size
        colors = make_colors(pts_xyz, labels, preds, args.moving_class)

        out_path = os.path.join(args.export_dir, f"frame_{idx:06d}.ply")
        export_ply(out_path, pts_xyz, colors)
        print("[INFO] wrote", out_path)


if __name__ == "__main__":
    main()

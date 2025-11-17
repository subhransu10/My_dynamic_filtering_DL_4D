import os
import argparse
import numpy as np
import torch
import MinkowskiEngine as ME

from .config import Config
from .dataset import RMOSNPZDataset
from .models import RMOSUNet

# WSL / headless: avoid hard-crash attempts
#since i had some crashes while working on wsl
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
    p.add_argument(
        "--data_root",
        type=str,
        default=Config.RMOS_ROOT,
        help="Root of preprocessed RMOS npz data",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint (best_moving.pth)",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Which split index list to use",
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start sample index within the chosen split",
    )
    p.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Optional end index (exclusive); default = end of split",
    )
    p.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size used during preprocessing (for scaling coords to meters)",
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
        help="Label index for 'moving' class in your setup",
    )
    p.add_argument(
        "--export_dir",
        type=str,
        default="./vis_seq",
        help=(
            "If interactive Open3D fails, we export per-frame PLYs here "
            "so you can browse them in an external viewer."
        ),
    )
    return p.parse_args()


def load_model(ckpt_path, num_classes, device, base_channels):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = RMOSUNet(
        in_channels=1,
        out_channels=num_classes,
        D=3,
        base_channels=base_channels,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_dataset(data_root, split):
    if split == "train":
        list_path = os.path.join(data_root, "train.txt")
    else:
        list_path = os.path.join(data_root, "val.txt")

    if not os.path.isfile(list_path):
        raise FileNotFoundError(f"Missing split index file: {list_path}")

    ds = RMOSNPZDataset(data_root, list_path)
    print(f"[INFO] {split} dataset size: {len(ds)}")
    return ds


def _to_me_coords(coords_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Normalize coords from dataset to MinkowskiEngine format.

    - If (N, 3): treat as (x,y,z) for a single sample -> add batch index via batched_coordinates.
    - If (N, 4): assume it's already (b,x,y,z).
    """
    if coords_np.ndim != 2:
        raise ValueError(f"coords must be 2D, got {coords_np.shape}")

    if coords_np.shape[1] == 3:
        # single frame, no batch -> wrap
        coords_t = torch.from_numpy(coords_np.astype(np.int32))
        from MinkowskiEngine.utils import batched_coordinates
        me_coords = batched_coordinates([coords_t])  # (N, 1+3)
    elif coords_np.shape[1] == 4:
        # already has batch index
        me_coords = torch.from_numpy(coords_np.astype(np.int32))
    else:
        raise ValueError(
            f"Unexpected coords shape {coords_np.shape}, expected (N,3) or (N,4)"
        )

    return me_coords.to(device)


@torch.no_grad()
def run_inference_for_index(dataset, idx, model, device):
    """
    Returns:
      pts_voxel (N,3) float32 in voxel coordinates,
      labels (N,) int64,
      preds  (N,) int64
    """
    item = dataset[idx]
    coords_np, feats_np, labels_np = item

    # To tensors
    coords_me = _to_me_coords(np.asarray(coords_np), device)
    feats = torch.as_tensor(feats_np, dtype=torch.float32, device=device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)

    # Sparse tensor
    x = ME.SparseTensor(feats, coordinates=coords_me, device=device)

    # Sanity: network expects D=3
    if x.D != 3:
        raise RuntimeError(f"SparseTensor dimension {x.D} != 3 (check coords format)")

    logits = model(x).F  # (N, C)
    preds = logits.argmax(1)

    # Strip batch index if present: coords_me is (N,4) -> use cols 1:4
    if coords_me.size(1) == 4:
        pts_voxel = coords_me[:, 1:4].to("cpu").numpy().astype(np.float32)
    else:
        pts_voxel = coords_me.to("cpu").numpy().astype(np.float32)

    labels = labels.to("cpu").numpy().astype(np.int64)
    preds = preds.to("cpu").numpy().astype(np.int64)

    return pts_voxel, labels, preds


def make_colors(pts, labels, preds, moving_class, mode=0):
    """
    mode 0: pred-vs-gt (TP/FP/FN highlighted)
    mode 1: pred only (moving vs static)
    mode 2: gt only (moving vs static)
    """
    n = pts.shape[0]
    colors = np.zeros((n, 3), dtype=np.float32)

    if n == 0:
        return colors

    if mode == 0:
        gt = labels
        pr = preds

        tp = (gt == moving_class) & (pr == moving_class)
        tn = (gt != moving_class) & (pr != moving_class)
        fp = (gt != moving_class) & (pr == moving_class)
        fn = (gt == moving_class) & (pr != moving_class)

        colors[tn] = [0.4, 0.4, 0.4]   # grey
        colors[tp] = [0.0, 0.8, 0.0]   # green
        colors[fp] = [1.0, 0.0, 0.0]   # red
        colors[fn] = [1.0, 1.0, 0.0]   # yellow

    elif mode == 1:
        pr = preds
        moving = pr == moving_class
        colors[~moving] = [0.4, 0.4, 0.4]
        colors[moving] = [1.0, 0.0, 0.0]  # red

    elif mode == 2:
        gt = labels
        moving = gt == moving_class
        colors[~moving] = [0.4, 0.4, 0.4]
        colors[moving] = [0.0, 0.0, 1.0]  # blue

    else:
        colors[:] = [0.5, 0.5, 0.5]

    return colors


def export_ply(path, pts, colors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)
    print(f"[EXPORT] Wrote {path}")


def run_interactive_viewer(args, dataset, model, device):
    if not HAS_O3D:
        raise RuntimeError("Open3D is not available in this environment.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(
        window_name="RMOS Viewer",
        width=1600,
        height=900,
        visible=True,
    ):
        vis.destroy_window()
        raise RuntimeError("Failed to create Open3D window.")

    start = max(args.start_index, 0)
    end = len(dataset) if args.end_index is None else min(args.end_index, len(dataset))
    if start >= end:
        vis.destroy_window()
        raise RuntimeError("Invalid start/end index range.")

    state = {
        "idx": start,
        "mode": 0,  # 0: pred-vs-gt, 1: pred only, 2: gt only
    }

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    def update_geometry():
        idx = state["idx"]
        pts_voxel, labels, preds = run_inference_for_index(dataset, idx, model, device)
        pts_vis = pts_voxel * args.voxel_size

        colors = make_colors(
            pts_vis,
            labels,
            preds,
            moving_class=args.moving_class,
            mode=state["mode"],
        )

        pcd.points = o3d.utility.Vector3dVector(pts_vis)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        center = pcd.get_center()
        pcd.translate(-center)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        print(
            f"[FRAME] idx={idx} | mode={state['mode']} "
            "(0=pred-vs-gt, 1=pred, 2=gt)"
        )

    def next_frame(_):
        if state["idx"] < end - 1:
            state["idx"] += 1
            update_geometry()
        return False

    def prev_frame(_):
        if state["idx"] > start:
            state["idx"] -= 1
            update_geometry()
        return False

    def toggle_color_mode(_):
        state["mode"] = (state["mode"] + 1) % 3
        update_geometry()
        return False

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(262, next_frame)  # Right arrow
    vis.register_key_callback(ord("B"), prev_frame)
    vis.register_key_callback(263, prev_frame)  # Left arrow
    vis.register_key_callback(ord("C"), toggle_color_mode)

    print(
        "\nControls:\n"
        "  → or N : next frame\n"
        "  ← or B : previous frame\n"
        "  C      : cycle color modes\n"
        "  Mouse  : rotate / zoom / pan\n"
    )

    update_geometry()
    vis.run()
    vis.destroy_window()


def run_export_sequence(args, dataset, model, device):
    """
    Fallback when Open3D interactive viewer is not possible on WSL2.
    Export a sequence of PLYs you can scroll through in an external viewer.
    """
    if not HAS_O3D:
        raise RuntimeError(
            "Open3D not available; cannot export PLY. Install open3d in this env."
        )

    start = max(args.start_index, 0)
    end = len(dataset) if args.end_index is None else min(args.end_index, len(dataset))
    if start >= end:
        raise RuntimeError("Invalid start/end index range.")

    os.makedirs(args.export_dir, exist_ok=True)
    print(
        f"[INFO] Interactive Open3D failed or disabled. "
        f"Exporting frames {start}..{end - 1} to: {args.export_dir}"
    )

    for idx in range(start, end):
        pts_voxel, labels, preds = run_inference_for_index(dataset, idx, model, device)
        pts_vis = pts_voxel * args.voxel_size
        colors = make_colors(
            pts_vis,
            labels,
            preds,
            moving_class=args.moving_class,
            mode=0,  # pred-vs-gt
        )
        out_path = os.path.join(args.export_dir, f"frame_{idx:06d}.ply")
        export_ply(out_path, pts_vis, colors)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = get_dataset(args.data_root, args.split)
    print(f"[INFO] Browsing indices {args.start_index} .. {len(dataset) - 1}")

    model = load_model(
        args.ckpt,
        num_classes=Config.NUM_CLASSES,
        device=device,
        base_channels=Config.BASE_CHANNELS,
    )

    if HAS_O3D:
        try:
            run_interactive_viewer(args, dataset, model, device)
            return
        except Exception as e:
            print("[WARN] Interactive Open3D failed:", repr(e))
            print("[WARN] Falling back to PLY sequence export mode.")
            run_export_sequence(args, dataset, model, device)
    else:
        print("[WARN] Open3D not installed; only export mode is available.")
        run_export_sequence(args, dataset, model, device)


if __name__ == "__main__":
    main()

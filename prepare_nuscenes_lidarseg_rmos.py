import os
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


MOVING_CLASS_NAMES = {
    # vehicles
    "vehicle.car",
    "vehicle.bus",
    "vehicle.truck",
    "vehicle.trailer",
    "vehicle.construction",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    # humans
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
    "human.pedestrian.stroller",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.construction_worker",
}


def voxelize(coords_xyz: np.ndarray, feats: np.ndarray, labels: np.ndarray, voxel_size: float):
    coords_scaled = coords_xyz / voxel_size
    coords_int = np.floor(coords_scaled).astype(np.int32)

    _, unique_inds = ME.utils.sparse_quantize(
        coordinates=coords_int,
        return_index=True,
    )

    coords_q = coords_int[unique_inds]
    feats_q  = feats[unique_inds]
    labels_q = labels[unique_inds]

    return coords_q.astype(np.int32), feats_q.astype(np.float32), labels_q.astype(np.int32)


def build_name_to_index(nusc: NuScenes):
    # Uses category.json (now lidarseg version) which contains "name" and "index".
    name_to_index = {}
    for c in nusc.category:
        # lidarseg uses plain names like "car", "vegetation", etc.
        name_to_index[c["name"]] = int(c["index"])
    return name_to_index


def make_binary_labels(seg_labels_u8: np.ndarray, moving_indices: set):
    # seg_labels_u8: (N,) uint8 indices
    labels = np.zeros(seg_labels_u8.shape[0], dtype=np.int32)
    if len(moving_indices) > 0:
        mask = np.isin(seg_labels_u8.astype(np.int32), np.array(sorted(list(moving_indices)), dtype=np.int32))
        labels[mask] = 1
    return labels


def main():
    missing = 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, required=True, help="nuScenes root (has samples/, sweeps/, v1.0-trainval/, lidarseg/)")
    ap.add_argument("--version", type=str, default="v1.0-trainval")
    ap.add_argument("--out_root", type=str, required=True, help="Output root for RMOS-style NPZ dataset")
    ap.add_argument("--voxel_size", type=float, default=0.1)
    ap.add_argument("--use_sweeps", type=int, default=0, help="0 = keyframes only; >0 optionally add sweeps later (not used here)")
    args = ap.parse_args()

    dataroot = args.dataroot
    out_root = Path(args.out_root)
    (out_root / "npz").mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=dataroot, verbose=False)

    name_to_index = build_name_to_index(nusc)
    print("Example category names:", list(name_to_index.keys())[:20])


    # Build the set of lidarseg indices that we consider "moving"
    moving_indices = set()
    for nm in MOVING_CLASS_NAMES:
        if nm in name_to_index:
            moving_indices.add(name_to_index[nm])

    print("Moving class indices:", sorted(list(moving_indices)))

    # We’ll create train/val split by official nuScenes split lists are a bit involved;
    # for now we do a simple deterministic split: every 5th scene -> val.
    scenes = nusc.scene
    scenes_sorted = sorted(scenes, key=lambda s: s["name"])
    val_scene_tokens = set([s["token"] for i, s in enumerate(scenes_sorted) if (i % 5) == 0])
    train_scene_tokens = set([s["token"] for s in scenes_sorted if s["token"] not in val_scene_tokens])

    train_list = []
    val_list = []

    for scene in tqdm(scenes_sorted, desc="Scenes"):
        scene_token = scene["token"]
        is_val = scene_token in val_scene_tokens

        # Walk samples in the scene
        sample_token = scene["first_sample_token"]
        frame_idx = 0

        # Use scene name as "seq" folder (safe for your Single-frame dataset)
        seq = scene["name"]  # e.g. "scene-0001"
        out_seq_dir = out_root / "npz" / seq
        out_seq_dir.mkdir(parents=True, exist_ok=True)

        while sample_token:
            sample = nusc.get("sample", sample_token)
            sd_token = sample["data"]["LIDAR_TOP"]
            sd = nusc.get("sample_data", sd_token)

            # Load points
            lidar_path = os.path.join(dataroot, sd["filename"])
            if not os.path.exists(lidar_path):
               missing += 1
               # Partial blobs: metadata references files not present.
               sample_token = sample["next"]
               frame_idx += 1
               continue

            pc = LidarPointCloud.from_file(lidar_path)  # shape (4, N)
            pts = pc.points.T  # (N, 4) -> x,y,z,intensity
            coords_xyz = pts[:, 0:3].astype(np.float32)
            intensity = pts[:, 3:4].astype(np.float32)

            # Load lidarseg labels aligned to this LIDAR_TOP keyframe
            seg = nusc.get("lidarseg", sd_token)
            seg_path = os.path.join(dataroot, seg["filename"])
            seg_labels = np.fromfile(seg_path, dtype=np.uint8)
            if seg_labels.shape[0] != coords_xyz.shape[0]:
                # rare mismatch, skip this frame
                sample_token = sample["next"]
                frame_idx += 1
                continue

            labels_bin = make_binary_labels(seg_labels, moving_indices)

            coords_q, feats_q, labels_q = voxelize(coords_xyz, intensity, labels_bin, args.voxel_size)

            out_name = f"{frame_idx:06d}.npz"
            out_path = out_seq_dir / out_name
            np.savez_compressed(out_path, coords=coords_q, feats=feats_q, labels=labels_q)

            rel = str(Path("npz") / seq / out_name)
            (val_list if is_val else train_list).append(rel)

            sample_token = sample["next"]
            frame_idx += 1

    (out_root / "train.txt").write_text("\n".join(train_list) + "\n")
    (out_root / "val.txt").write_text("\n".join(val_list) + "\n")

    print("Missing lidar files skipped:", missing)
    print("Done.")
    print("Train frames:", len(train_list))
    print("Val frames:", len(val_list))
    print("Output:", str(out_root))


if __name__ == "__main__":
    main()

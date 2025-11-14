import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME


class Config:
    # CHANGE this if needed, but you already have it set in rmos/config.py
    SEMANTICKITTI_ROOT = "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences"


# Official SemanticKITTI learning_map (reduced for training)
# From semantic-kitti.yaml (static/dynamic mapping)
LEARNING_MAP = {
    0: 0,      # "unlabeled"
    1: 0,      # "outlier"
    10: 1,     # "car"
    11: 1,
    13: 1,
    15: 1,
    16: 1,
    18: 1,
    20: 1,
    30: 2,     # "bicycle"
    31: 2,
    32: 2,
    40: 3,     # "motorcycle"
    44: 3,
    48: 3,
    49: 3,
    50: 4,     # "truck"
    51: 4,
    52: 4,
    60: 5,     # "other-vehicle"
    70: 6,     # "person"
    71: 6,
    72: 6,
    80: 7,     # "bicyclist"
    81: 7,
    99: 0,     # "other-static"
    252: 8,    # "road"
    253: 8,
    254: 8,
    255: 8,
    256: 8,
    257: 8,
    258: 8,
    259: 8,
    40_00: 9,  # just in case typo; safe no-op
}

# Fallback: any class not in LEARNING_MAP -> 0 (ignore / unlabeled)
MAX_RAW_ID = 260
LEARNING_MAP_ARRAY = np.zeros(MAX_RAW_ID + 1, dtype=np.int32)
for k, v in LEARNING_MAP.items():
    if k <= MAX_RAW_ID:
        LEARNING_MAP_ARRAY[k] = v


def load_points(bin_path: Path):
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    coords = pts[:, :3]  # x, y, z
    feats = pts[:, 3:4]  # intensity as single-channel feature
    return coords, feats


def load_labels(label_path: Path):
    lbl = np.fromfile(str(label_path), dtype=np.uint32).reshape(-1)
    # lower 16 bits are semantic label
    lbl = lbl & 0xFFFF
    # map to learning IDs (0 = "ignore/unlabeled" in our setting)
    lbl = np.where(lbl <= MAX_RAW_ID, LEARNING_MAP_ARRAY[lbl], 0)
    return lbl.astype(np.int32)


def voxelize(coords, feats, labels, voxel_size: float):
    """
    coords: (N, 3) float32 in meters
    feats:  (N, C)
    labels: (N,)
    """
    # scale to voxel grid
    coords_scaled = coords / voxel_size

    # MinkowskiEngine expects int coords; we floor
    coords_int = np.floor(coords_scaled).astype(np.int32)

    # Use sparse_quantize to get unique voxel indices
    # NOTE: NO batch index here; we work per-frame, so D=3.
    _, unique_inds = ME.utils.sparse_quantize(
        coordinates=coords_int,
        return_index=True,
    )

    coords_q = coords_int[unique_inds]
    feats_q = feats[unique_inds]
    labels_q = labels[unique_inds]

    assert (
        coords_q.shape[0] == feats_q.shape[0] == labels_q.shape[0]
    ), "Quantized coords/feats/labels mismatch"

    return coords_q, feats_q.astype(np.float32), labels_q.astype(np.int32)


def process_sequence(seq_id: str, root_dir: Path, out_root: Path, voxel_size: float):
    seq_dir = root_dir / seq_id
    velo_dir = seq_dir / "velodyne"
    label_dir = seq_dir / "labels"

    use_labels = label_dir.exists()

    frames = sorted(velo_dir.glob("*.bin"))

    out_npz_dir = out_root / "npz" / seq_id
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = []

    for bin_path in tqdm(frames, desc=f"Seq {seq_id}", unit="frame"):
        stem = bin_path.stem  # e.g. "000000"

        coords, feats = load_points(bin_path)

        if use_labels:
            lbl_path = label_dir / f"{stem}.label"
            if not lbl_path.exists():
                # If missing label: skip frame
                continue
            labels = load_labels(lbl_path)
            if labels.shape[0] != coords.shape[0]:
                # Corrupt / mismatched frame; skip
                continue
        else:
            # test sequences (no labels)
            labels = np.zeros(coords.shape[0], dtype=np.int32)

        coords_q, feats_q, labels_q = voxelize(coords, feats, labels, voxel_size)

        out_path = out_npz_dir / f"{stem}.npz"
        np.savez_compressed(
            out_path,
            coords=coords_q,
            feats=feats_q,
            labels=labels_q,
        )
        npz_paths.append(out_path)

    return npz_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semkitti_root",
        type=str,
        default=Config.SEMANTICKITTI_ROOT,
        help="Path to SemanticKITTI sequences directory (contains 00, 01, ...)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./data/rmos",
        help="Output root for RMOS npz + split files",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size in meters",
    )
    args = parser.parse_args()

    sem_root = Path(args.semkitti_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Train/Val split: standard SemanticKITTI style
    train_seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
    val_seqs = ["08"]

    train_files = []
    val_files = []

    # Process train sequences
    for seq in train_seqs:
        npz_paths = process_sequence(seq, sem_root, out_root, args.voxel_size)
        rel_paths = [
            str(p.relative_to(out_root)) for p in npz_paths
        ]  # store paths relative to out_root
        train_files.extend(rel_paths)

    # Process val sequences
    for seq in val_seqs:
        npz_paths = process_sequence(seq, sem_root, out_root, args.voxel_size)
        rel_paths = [
            str(p.relative_to(out_root)) for p in npz_paths
        ]
        val_files.extend(rel_paths)

    # Write split files
    with open(out_root / "train.txt", "w") as f:
        for p in train_files:
            f.write(p + "\n")

    with open(out_root / "val.txt", "w") as f:
        for p in val_files:
            f.write(p + "\n")

    print("Wrote:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Root:  {out_root}")


if __name__ == "__main__":
    main()

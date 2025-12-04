# added the temporal stacking which is the uniwue part of 4D dataset
import os
import re
import numpy as np
from torch.utils.data import Dataset


class RMOSNPZDataset(Dataset):
    """
    Single-frame dataset.
    Each line in list_path points to an NPZ under data_root.
    NPZ must contain: coords (N,3 int), feats (N,C float), labels (N,)
    """
    def __init__(self, data_root: str, list_path: str):
        super().__init__()
        self.data_root = data_root
        with open(list_path, "r") as f:
            self.files = [ln.strip() for ln in f.readlines() if ln.strip()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel = self.files[idx]
        npz_path = rel if os.path.isabs(rel) else os.path.join(self.data_root, rel)
        data = np.load(npz_path)

        coords = data["coords"].astype(np.int32)     # (N,3)
        feats  = data["feats"].astype(np.float32)    # (N,C)
        labels = data["labels"].astype(np.int32)     # (N,)

        return coords, feats, labels


def _parse_seq_and_frame(path: str):
    """
    Try to infer (seq, frame_id) from common SemanticKITTI/RMOS paths.
    Supports things like:
      npz/00/000123.npz
      npz/00_000123.npz
      .../sequences/00/velodyne/000123.npy
    """
    p = path.replace("\\", "/")

    # 1) parent folder as seq + filename as frame
    m = re.search(r"/(\d{2})/(\d{6})\.", p)
    if m:
        return m.group(1), int(m.group(2))

    # 2) seq_frame in basename
    m = re.search(r"(\d{2})[_\-](\d{6})", os.path.basename(p))
    if m:
        return m.group(1), int(m.group(2))

    return None, None


# rmos/dataset.py (ONLY the RMOSSequenceNPZDataset class)

class RMOSSequenceNPZDataset(Dataset):
    """
    Multi-frame temporal dataset for 4D MinkowskiEngine.
    We stack n_frames frames by appending a time coordinate t to coords.

    Output coords is (N_total, 4) int [x,y,z,t].
    Only current frame keeps labels; past frames labeled -1 (ignored in loss).

    time_feat:
      - "none": no extra time feature
      - "scalar": add a scalar time feature t_norm in [0,1]
    """
    def __init__(self, data_root: str, list_path: str,
                 n_frames: int = 3,
                 frame_stride: int = 1,
                 time_feat: str = "none"):
        super().__init__()
        assert n_frames >= 1
        assert time_feat in ("none", "scalar")
        self.data_root = data_root
        self.n_frames = n_frames
        self.frame_stride = max(1, frame_stride)
        self.time_feat = time_feat

        with open(list_path, "r") as f:
            self.files = [ln.strip() for ln in f.readlines() if ln.strip()]

        # Build a quick lookup for "same sequence previous frame"
        self.seq_frame_to_idx = {}
        for i, rel in enumerate(self.files):
            seq, fr = _parse_seq_and_frame(rel)
            if seq is not None:
                self.seq_frame_to_idx[(seq, fr)] = i

    def __len__(self):
        return len(self.files)

    def _load_npz(self, rel_path: str):
        npz_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.data_root, rel_path)
        data = np.load(npz_path)
        coords = data["coords"].astype(np.int32)
        feats  = data["feats"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        return coords, feats, labels

    def __getitem__(self, idx):
        cur_rel = self.files[idx]
        seq, fr = _parse_seq_and_frame(cur_rel)

        # Collect frame indices: current, prev1, prev2, ...
        frame_indices = [idx]
        if seq is not None and fr is not None:
            for k in range(1, self.n_frames):
                prev_fr = fr - k * self.frame_stride
                prev_idx = self.seq_frame_to_idx.get((seq, prev_fr), None)
                frame_indices.append(prev_idx if prev_idx is not None else idx)
        else:
            # Fall back to repeating current if parsing failed
            frame_indices += [idx] * (self.n_frames - 1)

        all_coords, all_feats, all_labels = [], [], []

        for t_i, fidx in enumerate(frame_indices):
            rel = self.files[fidx]
            coords3, feats, labels = self._load_npz(rel)

            # append time coordinate as integer
            tcol = np.full((coords3.shape[0], 1), t_i, dtype=np.int32)
            coords4 = np.concatenate([coords3, tcol], axis=1)  # (N,4)

            # OPTIONAL: append time feature (normalized)
            if self.time_feat != "none":
                if self.n_frames > 1:
                    t_norm = float(t_i) / float(max(1, self.n_frames - 1))
                else:
                    t_norm = 0.0
                t_feat = np.full((feats.shape[0], 1), t_norm, dtype=np.float32)
                feats = np.concatenate([feats, t_feat], axis=1)  # (N, C+1)

            # only current frame has labels, past frames ignored
            if t_i > 0:
                labels = np.full_like(labels, -1)  # ignore past frames

            all_coords.append(coords4)
            all_feats.append(feats)
            all_labels.append(labels)

        coords = np.concatenate(all_coords, axis=0)
        feats  = np.concatenate(all_feats, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return coords, feats, labels


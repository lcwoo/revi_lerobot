#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# DemoSpeedup: Create a new dataset with re-encoded videos containing only
# the downsampled (entropy-guided) frames. Parquet and meta are updated so
# the new dataset is self-contained (no SpeedupDatasetWrapper needed at train time).
"""
Create a speedup dataset: same as the entropy dataset but with videos re-encoded
to contain only the downsampled frames. Use this when you want a smaller
on-disk dataset and faster loading (no wrapper filtering at train time).

Requires: dataset with demo_dentropy feature (from compute_demo_dentropy_act or _diffusion).

Example:
  python -m lerobot.datasets.create_speedup_dataset \\
    --dataset_repo_id J-joon/habilis_alpha_shirt_dentropy \\
    --dataset_root /dev/shm/datasets/J-joon/habilis_alpha_shirt_dentropy \\
    --output_repo_id J-joon/habilis_alpha_shirt_speedup \\
    --output_root /dev/shm/datasets/J-joon/habilis_alpha_shirt_speedup \\
    --downsample_low_v 2 --downsample_high_v 4
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.speedup import (
    DEFAULT_DENTROPY_FEATURE,
    compute_speedup_indices,
)
from lerobot.datasets.utils import (
    DATA_DIR,
    EPISODES_DIR,
    VIDEO_DIR,
    load_info,
    load_episodes,
    write_info,
)
from lerobot.datasets.video_utils import (
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


def _scalar_int(x) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def _scalar_float(x) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def create_speedup_dataset(
    dataset: LeRobotDataset,
    output_dir: Path,
    *,
    label_key: str = DEFAULT_DENTROPY_FEATURE,
    low_v: int = 2,
    high_v: int = 4,
    threshold: float | None = None,
    video_backend: str | None = None,
    vcodec: str = "libsvtav1",
    tolerance_s: float = 1e-4,
) -> None:
    """
    Write a new dataset under output_dir with:
    - Data parquet: only rows at speedup indices; new global index and episode boundaries.
    - Meta/episodes: new length, dataset_from_index, dataset_to_index, videos/*/from_timestamp, to_timestamp, chunk_index, file_index (one video per episode).
    - Videos: re-encoded so each episode has one mp4 per camera containing only the kept frames.
    """
    dataset._ensure_hf_dataset_loaded()
    hf = dataset.hf_dataset
    root = Path(dataset.root)
    meta = dataset.meta
    fps = meta.fps
    video_keys = list(meta.video_keys) if meta.video_keys else []

    kept_indices = compute_speedup_indices(
        dataset, label_key=label_key, low_v=low_v, high_v=high_v, threshold=threshold
    )
    kept_set = set(kept_indices)
    n_total_new = len(kept_indices)

    # Episode boundaries in original dataset
    ep_col = hf["episode_index"]
    n = len(hf)
    ep_starts: dict[int, int] = {}
    ep_ends: dict[int, int] = {}
    for i in range(n):
        ep_idx = _scalar_int(ep_col[i])
        if ep_idx not in ep_starts:
            ep_starts[ep_idx] = i
        ep_ends[ep_idx] = i + 1

    # Per-episode: list of global indices kept
    ep_kept: dict[int, list[int]] = {}
    for ep_idx in sorted(ep_starts):
        from_i = ep_starts[ep_idx]
        to_i = ep_ends[ep_idx]
        ep_kept[ep_idx] = [i for i in range(from_i, to_i) if i in kept_set]

    # New episode boundaries (cumulative)
    new_ep_lengths: dict[int, int] = {ep: len(ep_kept[ep]) for ep in sorted(ep_kept)}
    cum = 0
    new_dataset_from: dict[int, int] = {}
    new_dataset_to: dict[int, int] = {}
    for ep in sorted(new_ep_lengths):
        new_dataset_from[ep] = cum
        cum += new_ep_lengths[ep]
        new_dataset_to[ep] = cum

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Info: copy and update total_frames
    info = load_info(root)
    info["total_frames"] = n_total_new
    write_info(info, output_dir)

    # Copy meta except episodes (we'll write new episodes)
    for item in (root / "meta").iterdir():
        if item.name == "episodes":
            continue
        dst = output_dir / "meta" / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    # Data: filter to kept rows, re-index and re frame_index, then save (preserves HF schema)
    filtered = hf.select(kept_indices)
    new_index_list = list(range(n_total_new))
    new_frame_index_list = []
    for ep_idx in sorted(ep_kept):
        new_frame_index_list.extend(range(len(ep_kept[ep_idx])))
    filtered = filtered.remove_columns(["index", "frame_index"])
    filtered = filtered.add_column("index", new_index_list)
    filtered = filtered.add_column("frame_index", new_frame_index_list)
    data_chunk_dir = output_dir / DATA_DIR / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(str(data_chunk_dir / "file-000.parquet"))

    # Episodes meta: one row per episode; one video file per episode (chunk_index=ep, file_index=0)
    episodes_meta = load_episodes(root)
    ep_columns = episodes_meta.column_names
    new_ep_rows = {col: [] for col in ep_columns}
    for ep_idx in sorted(new_ep_lengths):
        old_row = episodes_meta[ep_idx]
        new_len = new_ep_lengths[ep_idx]
        for col in ep_columns:
            val = old_row[col]
            if col == "length":
                val = np.int64(new_len)
            elif col == "dataset_from_index":
                val = np.int64(new_dataset_from[ep_idx])
            elif col == "dataset_to_index":
                val = np.int64(new_dataset_to[ep_idx])
            elif col == "data/chunk_index":
                val = np.int64(0)
            elif col == "data/file_index":
                val = np.int64(0)
            elif col.startswith("videos/") and col.endswith("/chunk_index"):
                val = np.int64(ep_idx)
            elif col.startswith("videos/") and col.endswith("/file_index"):
                val = np.int64(0)
            elif col.startswith("videos/") and col.endswith("/from_timestamp"):
                val = np.float64(0.0)
            elif col.startswith("videos/") and col.endswith("/to_timestamp"):
                val = np.float64((new_len - 1) / fps) if new_len else np.float64(0.0)
            new_ep_rows[col].append(val)
    # Build episodes table
    ep_table = pa.table({col: list(vals) for col, vals in new_ep_rows.items()})
    ep_dir = output_dir / EPISODES_DIR / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(ep_table, ep_dir / "file-000.parquet")

    # Videos: decode kept frames per episode and encode to new mp4
    if video_backend is None:
        video_backend = get_safe_default_codec()
    timestamps_col = hf["timestamp"]

    for ep_idx in tqdm(sorted(ep_kept), desc="Re-encoding videos"):
        kept_global = ep_kept[ep_idx]
        if not kept_global:
            continue
        ep = meta.episodes[ep_idx]
        # Timestamps of kept frames (relative to episode start for video PTS)
        first_ts = _scalar_float(timestamps_col[kept_global[0]])
        query_ts_rel = [_scalar_float(timestamps_col[g]) - first_ts for g in kept_global]

        for vid_key in video_keys:
            from_ts = ep[f"videos/{vid_key}/from_timestamp"]
            from_ts = _scalar_float(from_ts) if hasattr(from_ts, "item") else float(from_ts)
            shifted_ts = [from_ts + t for t in query_ts_rel]
            video_path_src = root / meta.get_video_file_path(ep_idx, vid_key)
            if not video_path_src.exists():
                continue
            frames = decode_video_frames(
                video_path_src, shifted_ts, tolerance_s, video_backend
            )
            # frames: (N, C, H, W) float [0,1]
            with tempfile.TemporaryDirectory(prefix="speedup_vid_") as tmpdir:
                tmpdir = Path(tmpdir)
                for i, fr in enumerate(frames):
                    if isinstance(fr, torch.Tensor):
                        fr = fr.cpu().numpy()
                    if fr.shape[0] == 3:
                        fr = fr.transpose(1, 2, 0)
                    fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(fr).save(tmpdir / f"frame-{i:06d}.png")
                out_video = (
                    output_dir / VIDEO_DIR / vid_key / f"chunk-{ep_idx:03d}" / "file-000.mp4"
                )
                out_video.parent.mkdir(parents=True, exist_ok=True)
                encode_video_frames(tmpdir, out_video, fps, vcodec=vcodec, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Create speedup dataset with re-encoded videos.")
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--output_repo_id", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--label_key", type=str, default=DEFAULT_DENTROPY_FEATURE)
    parser.add_argument("--downsample_low_v", type=int, default=2)
    parser.add_argument("--downsample_high_v", type=int, default=4)
    parser.add_argument("--vcodec", type=str, default="libsvtav1")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root is None:
        dataset_root = HF_LEROBOT_HOME / args.dataset_repo_id
    output_root = Path(args.output_root) if args.output_root else None
    if output_root is None:
        output_root = HF_LEROBOT_HOME / args.output_repo_id

    dataset = LeRobotDataset(args.dataset_repo_id, root=dataset_root)
    if args.label_key not in (dataset.meta.features or {}):
        raise ValueError(
            f"Dataset must have feature '{args.label_key}'. "
            "Run compute_demo_dentropy_act or compute_demo_dentropy_diffusion first."
        )

    create_speedup_dataset(
        dataset,
        output_root,
        label_key=args.label_key,
        low_v=args.downsample_low_v,
        high_v=args.downsample_high_v,
        vcodec=args.vcodec,
    )
    print(f"Speedup dataset written to {output_root}")


if __name__ == "__main__":
    main()

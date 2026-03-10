# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DemoSpeedup-style entropy-guided demonstration downsampling.

Uses per-frame entropy (e.g. demo_dentropy from proxy policy) to decide step size:
- High entropy (low confidence) -> smaller step (low_v) -> keep more frames
- Low entropy (high confidence) -> larger step (high_v) -> skip more frames
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Default feature name for entropy label (must match compute_demo_dentropy_*.py output)
DEFAULT_DENTROPY_FEATURE = "demo_dentropy"


def compute_speedup_indices_for_episode(
    entropy: np.ndarray,
    low_v: int,
    high_v: int,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Compute frame indices to keep for one episode (DemoSpeedup-style).

    High entropy (low confidence) -> step = low_v (denser sampling).
    Low entropy (high confidence) -> step = high_v (sparser sampling).

    Args:
        entropy: Per-frame entropy, shape (episode_len,).
        low_v: Step size when entropy is above threshold (keep more frames).
        high_v: Step size when entropy is below threshold (skip more frames).
        threshold: Entropy threshold. If None, use median(entropy).

    Returns:
        Indices to keep (relative to episode start), shape (n_kept,).
    """
    if entropy.size == 0:
        return np.array([], dtype=np.int64)
    if threshold is None:
        threshold = float(np.median(entropy))
    step_size = np.where(entropy > threshold, low_v, high_v).astype(np.int64)
    step_size = np.maximum(step_size, 1)

    indices = [0]
    i = 0
    L = len(entropy)
    while i < L - 1:
        step = int(step_size[i])
        next_i = min(i + step, L - 1)
        if next_i > i:
            indices.append(next_i)
        i = next_i

    return np.array(indices, dtype=np.int64)


def _scalar_int(x) -> int:
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def compute_speedup_indices(
    dataset: LeRobotDataset,
    label_key: str = DEFAULT_DENTROPY_FEATURE,
    low_v: int = 2,
    high_v: int = 4,
    threshold: float | None = None,
) -> list[int]:
    """
    Compute global dataset indices to keep for DemoSpeedup training.

    Uses the loaded hf_dataset (filtered or not). Episode boundaries are
    derived from hf_dataset["episode_index"] so indexing is correct for
    both full and episode-filtered datasets.

    Args:
        dataset: LeRobotDataset with demo_dentropy (or label_key) feature.
        label_key: Feature name for per-frame entropy.
        low_v: Step size in high-entropy (low confidence) segments.
        high_v: Step size in low-entropy (high confidence) segments.
        threshold: Per-episode threshold; None = median.

    Returns:
        List of global frame indices (in the current dataset's index space).
    """
    if hasattr(dataset, "_ensure_hf_dataset_loaded"):
        dataset._ensure_hf_dataset_loaded()
    hf = dataset.hf_dataset
    if hf is None:
        raise RuntimeError("Dataset must be loaded to compute speedup indices")
    if label_key not in hf.column_names:
        raise ValueError(
            f"Speedup requires feature '{label_key}'. "
            "Run compute_demo_dentropy_act or compute_demo_dentropy_diffusion first."
        )

    ep_col = hf["episode_index"]
    label = hf[label_key]
    n = len(hf)
    # Episode boundaries in current (possibly filtered) dataset index space
    ep_starts: dict[int, int] = {}
    ep_ends: dict[int, int] = {}
    for i in range(n):
        ep_idx = _scalar_int(ep_col[i])
        if ep_idx not in ep_starts:
            ep_starts[ep_idx] = i
        ep_ends[ep_idx] = i + 1

    all_indices: list[int] = []
    for ep_idx in sorted(ep_starts):
        from_i = ep_starts[ep_idx]
        to_i = ep_ends[ep_idx]
        ent = label[from_i:to_i]
        if hasattr(ent, "numpy"):
            ent = ent.numpy()
        ent = np.asarray(ent, dtype=np.float64)
        if ent.ndim >= 2:
            ent = ent.squeeze()
        entropy = ent.flatten()

        rel_indices = compute_speedup_indices_for_episode(
            entropy, low_v=low_v, high_v=high_v, threshold=threshold
        )
        global_indices = (from_i + rel_indices).tolist()
        all_indices.extend(global_indices)

    return all_indices


class SpeedupDatasetWrapper:
    """
    Wraps a LeRobotDataset to expose only DemoSpeedup downsampled frame indices.

    Training uses __len__ and __getitem__; other attributes (meta, features,
    episodes, etc.) are delegated to the base dataset.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        label_key: str = DEFAULT_DENTROPY_FEATURE,
        low_v: int = 2,
        high_v: int = 4,
        threshold: float | None = None,
    ):
        self._base = dataset
        self._indices = compute_speedup_indices(
            dataset, label_key=label_key, low_v=low_v, high_v=high_v, threshold=threshold
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        real_idx = self._indices[idx]
        return self._base[real_idx]

    @property
    def meta(self):
        return self._base.meta

    @property
    def features(self):
        return self._base.features

    @property
    def episodes(self):
        return self._base.episodes

    @property
    def num_frames(self) -> int:
        return len(self._indices)

    @property
    def num_episodes(self) -> int:
        return self._base.num_episodes

    def __repr__(self) -> str:
        return (
            f"SpeedupDatasetWrapper(base={self._base!r}, "
            f"num_frames={self.num_frames}, original_frames={self._base.num_frames})"
        )

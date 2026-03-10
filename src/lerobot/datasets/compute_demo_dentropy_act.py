from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES


def build_policy_and_preprocessor(
    policy_ckpt: Path, dataset_meta: LeRobotDatasetMetadata, device: torch.device
) -> Tuple[ACTPolicy, callable]:
    features = dataset_to_policy_features(dataset_meta.features)

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)

    policy = ACTPolicy.from_pretrained(policy_ckpt)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=dataset_meta.stats)
    return policy, preprocessor


def gaussian_kernel(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """
    DemoSpeedup/robobase/utils.py와 동일한 형태.
    Args:
        x: (batch_size, num_samples, dim)
    Returns:
        (batch_size, num_samples, num_samples)
    """
    batch_size, num_samples, dim = x.size()
    x_i = x.unsqueeze(2)  # (B, N, 1, D)
    x_j = x.unsqueeze(1)  # (B, 1, N, D)
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (B, N, N)
    kernel_values = torch.exp(-distances / (2 * bandwidth**2))
    return kernel_values


class KDE:
    """
    DemoSpeedup/robobase/utils.KDE 와 동일한 인터페이스.
    """

    def __init__(self, kde_flag: bool = True, marginal_flag: bool = True) -> None:
        self.flag = kde_flag
        self.marginal_flag = marginal_flag

    def kde_entropy(self, x: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, num_samples, dim)
        Returns:
            entropy: (batch_size,)
            max_density_points: (batch_size, dim)
        """
        batch_size, num_samples, dim = x.size()

        if self.flag:
            bandwidth = self.estimate_bandwidth(x[0])
            self.flag = False

        # DemoSpeedup 원본과 동일하게 최종적으로 1로 고정
        bandwidth = 1

        kernel_values = gaussian_kernel(x, bandwidth)  # (B, N, N)
        density = kernel_values.sum(dim=2) / num_samples  # (B, N)

        max_indices = torch.argmax(density, dim=1)  # (B,)
        batch_indices = torch.arange(batch_size, device=x.device)
        max_density_points = x[batch_indices, max_indices, :]  # (B, D)

        log_density = torch.log(density + 1e-8)
        entropy = -log_density.mean(dim=1, keepdim=True)  # (B, 1)

        return entropy.squeeze(-1), max_density_points

    def estimate_bandwidth(self, x: torch.Tensor, rule: str = "scott") -> float:
        num_samples, dim = x.size()
        std = x.std(dim=0).mean().item()

        if rule == "silverman":
            bandwidth = 1.06 * std * num_samples ** (-1 / 5)
        elif rule == "scott":
            bandwidth = std * num_samples ** (-1 / (dim + 4))
        else:
            raise ValueError("Unsupported rule. Choose 'silverman' or 'scott'.")

        return float(bandwidth)


def compute_dentropy_for_dataset(
    dataset: LeRobotDataset,
    policy: ACTPolicy,
    preprocessor,
    device: torch.device,
    num_mc_samples: int = 20,
    batch_size: int = 16,
    temporal_aggregation: bool = True,
) -> np.ndarray:
    """
    전체 frame 순서에 맞는 demo_dentropy 벡터를 계산.
    길이 = dataset.meta.total_frames

    temporal_aggregation=True (기본): 논문/공식 DemoSpeedup과 동일하게, 시점 t에서
    "t를 포함하는 모든 chunk"에서 나온 action 샘플을 모아 N*K 점으로 KDE (시간 방향 합침).
    """
    meta = dataset.meta
    total_frames = int(meta.total_frames)
    all_dentropy = np.zeros(total_frames, dtype=np.float32)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type != "cpu"),
    )

    policy.eval()
    kde = KDE()

    if temporal_aggregation:
        # buffer[t] = 시점 t를 포함하는 chunk들에서 나온 (N, D) 텐서들의 리스트
        buffer: List[List[torch.Tensor]] = [[] for _ in range(total_frames)]
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing demo_dentropy (pass 1: collect)"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                proc = preprocessor(batch)
                if policy.config.image_features:
                    proc = dict(proc)
                    proc[OBS_IMAGES] = [proc[key] for key in policy.config.image_features]

                mc_actions = []
                policy.train()
                for _ in range(num_mc_samples):
                    actions, _ = policy.model(proc)
                    mc_actions.append(actions)
                policy.eval()

                stacked = torch.stack(mc_actions, dim=0)  # (N, B, H, D)
                N, B, H, D = stacked.shape
                for b in range(B):
                    chunk_start = cursor + b
                    for h in range(H):
                        t = chunk_start + h
                        if t < total_frames:
                            # 시점 t에서 chunk_start에서 시작한 chunk의 step h → (N, D)
                            buffer[t].append(stacked[:, b, h, :].cpu())
                cursor += B

        # Pass 2: 시점별로 모은 샘플을 합쳐 KDE
        with torch.no_grad():
            for t in tqdm(range(total_frames), desc="Computing demo_dentropy (pass 2: KDE)"):
                if not buffer[t]:
                    continue
                # (num_chunks_t, N, D) -> (num_chunks_t * N, D)
                pooled = torch.cat(buffer[t], dim=0).to(device)
                pooled = pooled.unsqueeze(0)  # (1, total_samples, D)
                ent, _ = kde.kde_entropy(pooled)
                all_dentropy[t] = ent.item()
    else:
        # 기존 방식: (frame, step)마다 N점만 사용
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing demo_dentropy"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                proc = preprocessor(batch)
                if policy.config.image_features:
                    proc = dict(proc)
                    proc[OBS_IMAGES] = [proc[key] for key in policy.config.image_features]

                mc_actions = []
                policy.train()
                for _ in range(num_mc_samples):
                    actions, _ = policy.model(proc)
                    mc_actions.append(actions)
                policy.eval()

                stacked = torch.stack(mc_actions, dim=0)
                M, B, H, D = stacked.shape
                samples = stacked.permute(1, 2, 0, 3).reshape(B * H, M, D)
                ent_flat, _ = kde.kde_entropy(samples)
                ent = ent_flat.view(B, H)
                ent_flat_np = ent.reshape(-1).cpu().numpy().astype(np.float32)
                end = cursor + ent_flat_np.shape[0]
                if end > total_frames:
                    end = total_frames
                    ent_flat_np = ent_flat_np[: end - cursor]
                all_dentropy[cursor:end] = ent_flat_np
                cursor = end

    return all_dentropy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute demo_dentropy labels using a trained ACT proxy policy.")
    parser.add_argument(
        "--policy_ckpt",
        type=str,
        required=True,
        help="Path to ACTPolicy checkpoint directory (pretrained_model).",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="LeRobotDataset repo_id (e.g., J-joon/habilis_alpha_shirt_images).",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional root directory for dataset (if not using HF_LEROBOT_HOME).",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="New repo_id for labeled dataset.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional output root directory. Defaults to $HF_LEROBOT_HOME/output_repo_id.",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=20,
        help="Number of MC samples per state for entropy estimation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for offline labeling.",
    )
    parser.add_argument(
        "--no_temporal_aggregation",
        action="store_true",
        help="Disable temporal aggregation (use N samples per step only; default is to aggregate over chunks like paper).",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else None
    dataset_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=dataset_root)

    policy_ckpt = Path(args.policy_ckpt)
    policy, preprocessor = build_policy_and_preprocessor(policy_ckpt, dataset_meta, device)

    # ACT.model() 은 action shape (B, chunk_size, action_dim) 을 기대함. delta_timestamps 를 policy config 에 맞추면 데이터셋이 해당 chunk 를 반환함.
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )

    dentropy = compute_dentropy_for_dataset(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        device=device,
        num_mc_samples=args.num_mc_samples,
        batch_size=args.batch_size,
        temporal_aggregation=not args.no_temporal_aggregation,
    )

    # 새 dataset으로 demo_dentropy feature 추가
    output_root = Path(args.output_root) if args.output_root is not None else None
    add_features(
        dataset,
        features={
            "demo_dentropy": (
                dentropy,
                {"dtype": "float32", "shape": [1], "names": None},
            )
        },
        output_dir=output_root,
        repo_id=args.output_repo_id,
    )


if __name__ == "__main__":
    main()


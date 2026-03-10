from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _prepare_obs_images_for_stack
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def gaussian_kernel(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """
    DemoSpeedup/robobase/utils.py와 동일한 형태의 Gaussian kernel.

    Args:
        x: (batch_size, num_samples, dim)
        bandwidth: scalar bandwidth

    Returns:
        (batch_size, num_samples, num_samples)
    """
    batch_size, num_samples, dim = x.size()
    x_i = x.unsqueeze(2)  # (B, N, 1, D)
    x_j = x.unsqueeze(1)  # (B, 1, N, D)
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)
    kernel_values = torch.exp(-distances / (2 * bandwidth**2))
    return kernel_values


class KDE:
    """
    DemoSpeedup/robobase/utils.KDE 와 동일한 인터페이스의 KDE 엔트로피 추정기.
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


def build_policy_and_preprocessor_diffusion(
    policy_ckpt: Path, dataset_meta: LeRobotDatasetMetadata, device: torch.device
) -> Tuple[DiffusionPolicy, callable]:
    # DiffusionPolicy.from_pretrained 에서 config 를 함께 로드
    policy = DiffusionPolicy.from_pretrained(policy_ckpt)
    policy.to(device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(policy.config, dataset_stats=dataset_meta.stats)
    return policy, preprocessor


def _collect_cond_batch(proc, policy, OBS_STATE, OBS_ENV_STATE, OBS_IMAGES):
    """Preprocess batch into cond_batch for DiffusionPolicy.generate_actions."""
    if OBS_STATE not in proc:
        raise RuntimeError("Preprocessed batch 에 'observation.state' 가 없습니다.")
    state = proc[OBS_STATE]
    if state.ndim == 2:
        state_seq = state.unsqueeze(1).repeat(1, policy.config.n_obs_steps, 1)
    elif state.ndim == 3:
        B, T, D = state.shape
        if T >= policy.config.n_obs_steps:
            state_seq = state[:, : policy.config.n_obs_steps]
        else:
            pad = state[:, -1:].repeat(1, policy.config.n_obs_steps - T, 1)
            state_seq = torch.cat([state, pad], dim=1)
    else:
        raise RuntimeError(f"Unsupported state shape for diffusion: {state.shape}")
    cond_batch: dict = {OBS_STATE: state_seq}
    if getattr(policy.config, "env_state_feature", None) and OBS_ENV_STATE in proc:
        env = proc[OBS_ENV_STATE]
        if env.ndim == 2:
            env_seq = env.unsqueeze(1).repeat(1, policy.config.n_obs_steps, 1)
        elif env.ndim == 3:
            B_e, T_e, D_e = env.shape
            if T_e >= policy.config.n_obs_steps:
                env_seq = env[:, : policy.config.n_obs_steps]
            else:
                pad_e = env[:, -1:].repeat(1, policy.config.n_obs_steps - T_e, 1)
                env_seq = torch.cat([env, pad_e], dim=1)
        else:
            raise RuntimeError(f"Unsupported env_state shape for diffusion: {env.shape}")
        cond_batch[OBS_ENV_STATE] = env_seq
    if policy.config.image_features:
        image_keys = list(policy.config.image_features)
        for key in image_keys:
            if key not in proc:
                raise RuntimeError(f"Preprocessed batch 에 이미지 키 '{key}' 가 없습니다.")
            img = proc[key]
            if policy.config.n_obs_steps == 1 and img.ndim == 4:
                img = img.unsqueeze(1)
            cond_batch[key] = img
        _prepare_obs_images_for_stack(cond_batch, image_keys, policy.config.crop_shape)
        cond_batch[OBS_IMAGES] = torch.stack([cond_batch[key] for key in image_keys], dim=-4)
    return cond_batch


def compute_dentropy_for_dataset_diffusion(
    dataset: LeRobotDataset,
    policy: DiffusionPolicy,
    preprocessor,
    device: torch.device,
    num_mc_samples: int = 20,
    batch_size: int = 16,
    temporal_aggregation: bool = True,
) -> np.ndarray:
    """
    DiffusionPolicy 기반 proxy policy 로 DemoSpeedup-style KDE entropy 를 계산.

    temporal_aggregation=True (기본): 논문/공식 DemoSpeedup과 동일하게, 시점 t에서
    "t를 포함하는 모든 chunk"에서 나온 action 샘플을 모아 N*K 점으로 KDE (시간 방향 합침).
    """
    meta = dataset.meta
    total_frames = int(meta.total_frames)
    all_dentropy = np.zeros(total_frames, dtype=np.float32)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type != "cpu"))
    policy.eval()
    kde = KDE()

    if temporal_aggregation:
        buffer: List[List[torch.Tensor]] = [[] for _ in range(total_frames)]
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing demo_dentropy (pass 1: collect)"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                proc = preprocessor(batch)
                cond_batch = _collect_cond_batch(proc, policy, OBS_STATE, OBS_ENV_STATE, OBS_IMAGES)
                B = cond_batch[OBS_STATE].shape[0]

                mc_actions = []
                for _ in range(num_mc_samples):
                    actions = policy.diffusion.generate_actions(cond_batch)
                    mc_actions.append(actions)
                stacked = torch.stack(mc_actions, dim=0)  # (N, B, H, D)
                N, B, H, D = stacked.shape
                for b in range(B):
                    chunk_start = cursor + b
                    for h in range(H):
                        t = chunk_start + h
                        if t < total_frames:
                            buffer[t].append(stacked[:, b, h, :].cpu())
                cursor += B

        with torch.no_grad():
            for t in tqdm(range(total_frames), desc="Computing demo_dentropy (pass 2: KDE)"):
                if not buffer[t]:
                    continue
                pooled = torch.cat(buffer[t], dim=0).to(device).unsqueeze(0)
                ent, _ = kde.kde_entropy(pooled)
                all_dentropy[t] = ent.item()
    else:
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing demo_dentropy (diffusion)"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                proc = preprocessor(batch)
                cond_batch = _collect_cond_batch(proc, policy, OBS_STATE, OBS_ENV_STATE, OBS_IMAGES)

                mc_actions = []
                for _ in range(num_mc_samples):
                    actions = policy.diffusion.generate_actions(cond_batch)
                    mc_actions.append(actions)
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
    parser = argparse.ArgumentParser(
        description="Compute demo_dentropy labels using a trained DiffusionPolicy proxy (DemoSpeedup-style KDE)."
    )
    parser.add_argument(
        "--policy_ckpt",
        type=str,
        required=True,
        help="Path to DiffusionPolicy checkpoint directory (pretrained_model).",
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

    policy_ckpt = Path(args.policy_ckpt).resolve()
    if not policy_ckpt.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory not found: {policy_ckpt}")
    policy, preprocessor = build_policy_and_preprocessor_diffusion(policy_ckpt, dataset_meta, device)

    # delta_timestamps 로 observation/action chunk 를 policy 에 맞게 로드 (ACT 와 동일)
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )

    dentropy = compute_dentropy_for_dataset_diffusion(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        device=device,
        num_mc_samples=args.num_mc_samples,
        batch_size=args.batch_size,
        temporal_aggregation=not args.no_temporal_aggregation,
    )

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


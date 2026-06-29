import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    """Stack the pairs into batched tensors for the DataLoader."""
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


@dataclass
class MainDataset(Dataset):
    """Per-frame calibrator dataset. `initialize(indices)` scans `dataset_path` in
    sorted order and builds a per-frame index map for the trajectories selected by
    `indices`. `__getitem__` then gathers one frame's precomputed similarity features
    from disk."""

    dataset_path: str | None = None
    epoch_size_divisor: int = 3
    random_sampling: bool = False
    iou_threshold: float = 0.5
    easy_positive_iou: float = 1.0           # frames with iou > this are "easy positives"
    easy_positive_factor: float = 1.0        # < 1 down-samples, > 1 upsamples easy positives
    hard_trajectory_factor: float = 1.0      # > 1 upsamples frames of below-median-oracle-coverage trajectories

    _frame_map: list = field(default_factory=list)

    def initialize(self, indices):
        """Build the per-frame index map for the trajectories at `indices` (positions
        into the sorted listing of `dataset_path`).

        When `hard_trajectory_factor != 1.0`, the median ORACLE coverage across the
        selected trajectories is used as the split point — trajectories below the
        median are 'hard' and their frames get re-weighted by `hard_trajectory_factor`."""

        clean_directory = Path(self.dataset_path)
        clean_paths = sorted(clean_directory / filename for filename in os.listdir(clean_directory))
        selected_paths = [clean_paths[trajectory_idx] for trajectory_idx in indices]

        # First pass — per-trajectory entries + oracle coverage
        per_trajectory = []
        for path in selected_paths:
            entries, oracle_coverage = self._build_entries_with_coverage(path)
            per_trajectory.append((entries, oracle_coverage))

        # Compute median oracle coverage to use as the hard/easy split
        finite_covs = np.array([cov for _, cov in per_trajectory if not np.isnan(cov)])
        if self.hard_trajectory_factor != 1.0 and len(finite_covs) > 0:
            median_coverage = float(np.median(finite_covs))
        else:
            median_coverage = -1.0   # disables the hard/easy split

        # Second pass — apply per-frame multiplier (composes hard-trajectory and
        # easy-positive factors).
        self._frame_map = []
        for entries, oracle_coverage in per_trajectory:
            is_hard = not np.isnan(oracle_coverage) and oracle_coverage < median_coverage
            hard_factor = self.hard_trajectory_factor if is_hard else 1.0
            rng = np.random.default_rng(seed=hash(entries[0][0]) & 0xFFFFFFFF)
            for entry in entries:
                _, _, iou_float = entry
                easy_factor = self.easy_positive_factor if iou_float > self.easy_positive_iou else 1.0
                factor = hard_factor * easy_factor
                if factor == 1.0:
                    self._frame_map.append(entry)
                    continue
                if factor < 1.0:
                    if rng.random() < factor:
                        self._frame_map.append(entry)
                else:
                    n_copies = int(factor)
                    if rng.random() < (factor - n_copies):
                        n_copies += 1
                    for _ in range(n_copies):
                        self._frame_map.append(entry)

    def _build_entries_with_coverage(self, path):
        """Load one trajectory pickle and return (base_entries, oracle_coverage).
        Oracle coverage is mean(iou > iou_threshold) over non-occluded frames."""

        experiment = pickle.load(open(path, "rb"))
        iou_array = np.asarray(experiment.iou_scores, dtype=np.float32)
        occ_array = np.asarray(experiment.occlusions, dtype=np.float32)
        not_occluded = occ_array < 0.5
        if not_occluded.any():
            oracle_coverage = float((iou_array[not_occluded] > self.iou_threshold).mean())
        else:
            oracle_coverage = float("nan")
        entries = [(str(path), int(frame_idx), float(iou))
                          for frame_idx, iou in enumerate(iou_array)]
        return entries, oracle_coverage

    def _load_features(self, path):
        """Load one trajectory's feature tensor from disk."""

        return pickle.load(open(path, "rb")).features

    def __len__(self):
        """Reduces the number of samples per epoch to control the training and evaluation time."""

        return len(self._frame_map) // self.epoch_size_divisor

    def __getitem__(self, idx):
        """Load one frame's full precomputed feature tensor (all channels) from disk and its label."""

        if self.random_sampling:
            sample_idx = int(np.random.randint(0, len(self._frame_map)))
        else:
            sample_idx = idx * self.epoch_size_divisor

        path, frame_idx, iou = self._frame_map[sample_idx]

        feature = self._load_features(path)[frame_idx].astype(np.float32)
        label = 1.0 if iou > self.iou_threshold else 0.0
        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.float32)

    def __getitems__(self, indices):
        """Batched fetch hook for PyTorch's DataLoader."""

        return [self.__getitem__(i) for i in indices]

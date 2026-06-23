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
    """Per-frame calibrator dataset. `initialize(indices)` scans `dataset_path` in sorted
    order and builds a per-frame index map for the trajectories selected by `indices`.
    `__getitem__` then gathers one frame's precomputed similarity features from disk."""

    dataset_path: str | None = None
    epoch_size_divisor: int = 3
    random_sampling: bool = False
    iou_threshold: float = 0.5

    _frame_map: list = field(default_factory=list)

    def initialize(self, indices):
        """Build the per-frame index map for the trajectories at `indices` (positions into
        the listing of `dataset_path`)."""

        clean_directory = Path(self.dataset_path)
        trajectory_paths = [clean_directory / filename for filename in os.listdir(clean_directory)]

        self._frame_map = []
        for trajectory_idx in indices:
            self._frame_map.extend(self._index_clean(trajectory_paths[trajectory_idx]))

    def _index_clean(self, clean_path):
        """Per-frame `(clean_path, frame_idx, true_iou)` entries for one trajectory."""

        experiment = pickle.load(open(clean_path, "rb"))
        return [(str(clean_path), frame_idx, float(iou))
                  for frame_idx, iou in enumerate(experiment.iou_scores)]

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

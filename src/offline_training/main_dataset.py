import os
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    """Stack `(feature, label)` pairs into batched tensors. When `__getitems__` already returns
    a pre-gathered `(features, labels)` batch, pass it straight through."""

    if isinstance(batch, tuple) and len(batch) == 2 and torch.is_tensor(batch[0]):
        return batch
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


@dataclass
class MainDataset(Dataset):
    """Per-frame calibrator dataset.

    `initialize(indices)` reads the trajectory pickles at `indices` and stacks every frame's
    precomputed similarity features and its `iou > iou_threshold` label into one tensor, held
    on the GPU when CUDA is available. All the I/O happens there, so `__getitem__` is a pure
    in-memory slice with no per-frame disk read."""

    dataset_path: str | None = None
    iou_threshold: float = 0.5

    _features: torch.Tensor | None = None      # (N, 1, H, W, C) float32
    _labels: torch.Tensor | None = None        # (N,) float32

    def initialize(self, indices):
        """Load the trajectories at `indices` (positions into the sorted listing of
        `dataset_path`) and stack all their frames into the feature and label tensors."""

        directory = Path(self.dataset_path)
        paths = sorted(directory / filename for filename in os.listdir(directory))

        features, labels = [], []
        for index in indices:
            experiment = pickle.load(open(paths[index], "rb"))
            features.append(np.asarray(experiment.features, dtype=np.float32))
            labels.append(np.asarray(experiment.iou_scores, dtype=np.float32) > self.iou_threshold)

        stacked_features = np.concatenate(features, axis=0)
        stacked_labels = np.concatenate(labels, axis=0).astype(np.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._features = torch.from_numpy(stacked_features).to(device)
        self._labels = torch.from_numpy(stacked_labels).to(device)

    def __len__(self):
        """Number of frames in the dataset, one training sample per frame."""

        return len(self._features)

    def __getitem__(self, idx):
        """Return one frame's feature tensor (all channels) and its label."""

        return self._features[idx], self._labels[idx]

    def __getitems__(self, indices):
        """Batched fetch hook for PyTorch's DataLoader: gather the whole batch in one indexing
        op and return a pre-collated `(features, labels)` tuple consumed by `collate_fn`."""

        selection = torch.as_tensor(indices, device=self._features.device)
        return self._features[selection], self._labels[selection]

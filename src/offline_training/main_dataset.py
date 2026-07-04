import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    """Stack `(feature, label)` pairs into batched tensors. When the dataset's
    `__getitems__` already returns a pre-gathered `(features, labels)` batch (the
    GPU-resident path), pass it straight through."""

    if isinstance(batch, tuple) and len(batch) == 2 and torch.is_tensor(batch[0]):
        return batch
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


@dataclass
class MainDataset(Dataset):
    """Per-frame calibrator dataset.

    `initialize(indices)` lists the trajectory pickles under `dataset_path` (sorted),
    keeps those at `indices`, and builds a flat per-frame index map. `__getitem__` returns
    one frame's precomputed similarity features and its `iou > iou_threshold` label.

    When CUDA is available and `load_on_gpu` is set, every selected frame is stacked into a
    single GPU tensor at `initialize` time, so training has zero per-batch host->device
    transfer (the bottleneck for this small model)."""

    dataset_path: str | None = None
    epoch_size_divisor: int = 3
    random_sampling: bool = False
    iou_threshold: float = 0.5
    load_on_gpu: bool = True

    _frame_map: list = field(default_factory=list)          # [(path, frame_idx, iou), ...]
    _feature_cache: dict = field(default_factory=dict)      # path -> in-RAM features array
    _gpu_features: object = None                            # (N, 1, H, W, C) float32 on GPU, or None
    _gpu_labels: object = None                              # (N,) float32 on GPU, or None

    # -------------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------------

    def initialize(self, indices):
        """Build the per-frame index map for the trajectories at `indices` (positions into
        the sorted listing of `dataset_path`), then optionally preload it onto the GPU."""

        self._feature_cache = {}
        clean_directory = Path(self.dataset_path)
        clean_paths = sorted(clean_directory / filename for filename in os.listdir(clean_directory))
        selected_paths = [clean_paths[trajectory_idx] for trajectory_idx in indices]

        self._frame_map = []
        for path in selected_paths:
            self._frame_map.extend(self._build_entries(path))

        self._gpu_features = None
        self._gpu_labels = None
        if self.load_on_gpu and torch.cuda.is_available():
            self._preload_to_gpu()

    def _build_entries(self, path):
        """Load one trajectory pickle, cache its feature tensor in RAM (one read per
        trajectory, so `__getitem__` never re-reads the pickle per frame), and return its
        per-frame `(path, frame_idx, iou)` entries."""

        experiment = pickle.load(open(path, "rb"))
        self._feature_cache[str(path)] = np.asarray(experiment.features)
        iou_scores = np.asarray(experiment.iou_scores, dtype=np.float32)
        return [(str(path), int(frame_idx), float(iou)) for frame_idx, iou in enumerate(iou_scores)]

    def _preload_to_gpu(self):
        """Stack every frame in `_frame_map` into one GPU tensor (features + labels) and drop
        the CPU cache. After this, `__getitem__`/`__getitems__` return GPU slices."""

        sample_path, sample_frame, _ = self._frame_map[0]
        sample = self._feature_cache[sample_path][sample_frame]

        features = np.empty((len(self._frame_map), *sample.shape), dtype=np.float32)
        labels = np.empty(len(self._frame_map), dtype=np.float32)
        for index, (path, frame_idx, iou) in enumerate(self._frame_map):
            features[index] = self._feature_cache[path][frame_idx]
            labels[index] = 1.0 if iou > self.iou_threshold else 0.0

        self._gpu_features = torch.from_numpy(features).cuda()
        self._gpu_labels = torch.from_numpy(labels).cuda()
        self._feature_cache = {}       # free CPU RAM; the data now lives on the GPU

    def _load_features(self, path):
        """Return a trajectory's feature tensor from the in-RAM cache (populated in
        `initialize`), falling back to a disk read if it isn't cached."""

        cached = self._feature_cache.get(str(path))
        if cached is not None:
            return cached
        return pickle.load(open(path, "rb")).features

    # -------------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------------

    def __len__(self):
        """Samples per epoch — divided down by `epoch_size_divisor` to bound epoch time."""

        return len(self._frame_map) // self.epoch_size_divisor

    def _sample_index(self, idx):
        """Map a DataLoader index to a `_frame_map` index (random when `random_sampling`)."""

        if self.random_sampling:
            return int(np.random.randint(0, len(self._frame_map)))
        return idx * self.epoch_size_divisor

    def __getitem__(self, idx):
        """Return one frame's feature tensor (all channels) and its label."""

        sample_idx = self._sample_index(idx)
        if self._gpu_features is not None:
            return self._gpu_features[sample_idx], self._gpu_labels[sample_idx]

        path, frame_idx, iou = self._frame_map[sample_idx]
        feature = self._load_features(path)[frame_idx].astype(np.float32)
        label = 1.0 if iou > self.iou_threshold else 0.0
        return torch.from_numpy(feature), torch.tensor(label, dtype=torch.float32)

    def __getitems__(self, indices):
        """Batched fetch hook for PyTorch's DataLoader.

        GPU-resident path: gather the WHOLE batch in a single indexing op and return a
        pre-collated `(features, labels)` tuple (passed through by `collate_fn`), avoiding
        one CUDA index kernel per sample — which otherwise dominates for this small model."""

        if self._gpu_features is None:
            return [self.__getitem__(i) for i in indices]

        count = len(self._frame_map)
        if self.random_sampling:
            selection = torch.randint(0, count, (len(indices),), device=self._gpu_features.device)
        else:
            selection = torch.as_tensor([i * self.epoch_size_divisor for i in indices],
                                        device=self._gpu_features.device)
        return self._gpu_features[selection], self._gpu_labels[selection]

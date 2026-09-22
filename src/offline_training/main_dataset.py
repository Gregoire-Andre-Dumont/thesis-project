"""The calibrator's training set: one sample per PROPOSAL, labelled with that proposal's true mask IoU.

SAM emits three competing masks a frame and the tracker keeps one. All three are samples -- the two it
rejected are the only examples of a bad mask the dataset holds. The target is the IoU itself, not a
pass/fail flag, because ranking three masks needs an ordering and a thresholded label carries none.
"""
import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset


def collate_fn(batch):
    """Stack `(feature, label)` pairs; pass a pre-gathered batch from `__getitems__` straight through."""

    if isinstance(batch, tuple) and len(batch) == 2 and torch.is_tensor(batch[0]):
        return batch
    features, labels = zip(*batch)
    return torch.stack(features), torch.stack(labels)


@dataclass
class MainDataset(Dataset):
    """Per-proposal calibrator dataset, indexed by TRAJECTORY.

    `dataset_path` holds one subfolder per memory-corruption probability (`p0.00`, `p0.05`, ...), each with
    the same trajectories rolled out at that rate; `probabilities` picks which to draw from. An index
    addresses a trajectory, not a file, and pulls it from every selected folder at once -- so a clip's
    near-identical copies cannot be split across a train/validation boundary.

    `initialize(indices)` stacks those trajectories into CPU tensors. They stay on the host so the loader's
    workers can prefetch batches -- a worker cannot touch a CUDA tensor, so GPU-resident storage forces
    `num_workers: 0`. The training loop moves each batch to the device itself."""

    dataset_path: str | None = None
    probabilities: list[float] = field(default_factory=lambda: [0.0])

    # Keep frames where the target is OCCLUDED, labelled 0. At deployment the calibrator scores every frame,
    # occluded ones included, so excluding them trains it on a distribution it will not meet.
    include_occluded: bool = False

    # Which per-proposal IoU to regress. The two arrays in the pickle are NOT the same contest: they
    # correlate 0.91 but rank the three proposals differently on a third of frames.
    #   proposal_true_iou -- box IoU against the GT box. What the oracles select on (`use_mask_iou: False`)
    #                        and what coverage is measured in, so it is the SELECTOR's objective.
    #   iou_scores        -- mask IoU against a box-prompted pseudo-GT mask. What the gate was calibrated
    #                        on, kept as the default so a change here cannot silently move the gate.
    label: str = "iou_scores"

    # Per-proposal scalars carried ALONGSIDE the similarity map, for a controller that should see what SAM
    # itself thinks as well as what the anchor looks like:
    #   proposal_iou_scores -- SAM's IoU token for that proposal, its own mask-quality estimate
    #   object_score        -- SAM's object-presence logit for the frame, shared by all three proposals
    # They ride as constant extra channels on the map rather than a second tensor, so the loader, the
    # collate function and the trainer are untouched; `CNNFixed` splits them off before the convolutions
    # and feeds them straight to its head. Empty by default, so the gate is unaffected.
    scalars: tuple[str, ...] = ()

    _features: torch.Tensor | None = None     # (samples, 1, grid, grid, channels) float32
    _labels: torch.Tensor | None = None       # (samples,) float32, the proposal's true IoU under `label`

    # ---------------------------------------------------------------------------------------
    # what the dataset is made of
    # ---------------------------------------------------------------------------------------

    def folders(self):
        """The corruption folders this dataset draws from, in the order given."""

        return [Path(self.dataset_path) / f"p{float(probability):.2f}" for probability in self.probabilities]

    def trajectories(self):
        """Trajectory stems present in EVERY selected folder, sorted -- what indices address.

        Intersecting rather than unioning matters while the dataset is still being written: a trajectory
        finished at one corruption level but not another would otherwise be drawn inconsistently."""

        listings = [{Path(name).stem for name in os.listdir(folder)} for folder in self.folders()]
        return sorted(set.intersection(*listings)) if listings else []

    def labelled(self, experiment):
        """Frames carrying a real label.

        A VISIBLE frame with an annotated box is labelled with its true IoU. An OCCLUDED frame is labelled 0
        -- correct, since the target is absent -- and is kept only when `include_occluded` is set.

        An UNANNOTATED frame is excluded either way: no box and not marked occluded means the target may
        well be there and simply was not labelled, so its zero would teach that a good mask deserves
        nothing. Those annotation gaps are a fifth of this dataset's apparent occlusions."""

        occlusions = np.asarray(experiment.occlusions, float)
        has_box = np.asarray(experiment.true_bboxes)[:, 2] > 0
        visible = (occlusions < 0.5) & has_box
        return visible | (occlusions > 0.5) if self.include_occluded else visible

    def scorable(self, experiment):
        """The evaluation subset: labelled frames from the first occlusion on, minus the anchor.

        Narrower than what the model trains on. The anchor is the reference every similarity map is measured
        against, so its crop scores a perfect match; and post-occlusion is the regime the claim is about."""

        keep = self.labelled(experiment).copy()
        occlusions = np.asarray(experiment.occlusions, float)
        occluded = occlusions > 0.5
        first_occlusion = int(np.argmax(occluded)) if occluded.any() else len(occlusions)
        keep[:first_occlusion] = False
        keep[0] = False
        return keep

    # ---------------------------------------------------------------------------------------
    # torch Dataset
    # ---------------------------------------------------------------------------------------

    def initialize(self, indices):
        """Load the trajectories at `indices` from every selected folder and stack their proposals."""

        stems = self.trajectories()
        features, labels = [], []
        for index in indices:
            for folder in self.folders():
                experiment = pickle.load(open(folder / f"{stems[index]}.pkl", "rb"))
                iou_scores = np.asarray(getattr(experiment, self.label), dtype=np.float32)
                if iou_scores.ndim != 2:              # single-proposal pickle from an older schema
                    continue
                keep = self.labelled(experiment)
                if not keep.any():
                    continue

                # (frames, proposals, grid, grid, channels) -> one sample per proposal, in label order.
                similarity_maps = np.asarray(experiment.features, dtype=np.float32)[keep]
                per_proposal = similarity_maps.reshape(-1, 1, *similarity_maps.shape[2:])
                per_proposal = self.with_scalars(per_proposal, experiment, keep, similarity_maps.shape[1])
                features.append(per_proposal)
                labels.append(iou_scores[keep].reshape(-1))

        self._features = torch.from_numpy(np.concatenate(features, axis=0))
        self._labels = torch.from_numpy(np.concatenate(labels, axis=0))

    def with_scalars(self, per_proposal, experiment, keep, n_proposals):
        """Append each requested scalar as a constant channel on every proposal's map.

        A per-frame scalar (`object_score`) is repeated across the frame's proposals; a per-proposal one
        (`proposal_iou_scores`) is taken as it stands. Constant planes are wasteful in the convolutions,
        which is why `CNNFixed` slices them off before them -- the point is only that one tensor still
        carries everything, so nothing downstream has to learn about a second input."""

        if not self.scalars:
            return per_proposal

        columns = []
        for name in self.scalars:
            values = np.asarray(getattr(experiment, name), dtype=np.float32)
            if values.ndim == 1:                          # per-frame: every proposal sees the same value
                values = np.repeat(values[:, None], n_proposals, axis=1)
            columns.append(values[keep].reshape(-1))

        stacked = np.stack(columns, axis=-1)              # (samples, n_scalars)
        planes = np.broadcast_to(stacked[:, None, None, None, :],
                                 per_proposal.shape[:4] + (len(self.scalars),))
        return np.concatenate([per_proposal, planes.astype(np.float32)], axis=-1)

    def __len__(self):
        """Number of proposal samples: three per labelled frame, per corruption level."""

        return len(self._features)

    def __getitem__(self, index):
        """One proposal's similarity map and its true IoU."""

        return self._features[index], self._labels[index]

    def __getitems__(self, indices):
        """Batched fetch for the DataLoader: one indexing op, pre-collated for `collate_fn`."""

        selection = torch.as_tensor(indices)
        return self._features[selection], self._labels[selection]

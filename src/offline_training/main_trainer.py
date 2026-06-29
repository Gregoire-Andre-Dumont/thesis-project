"""Module for example training block."""

import contextlib
import gc

import wandb
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from tqdm import tqdm

from src.utils.trainer_logger import Logger
from epochalyst.training.torch_trainer import TorchTrainer
from epochalyst.training.utils import batch_to_device


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for training the memory control of DSAM 2."""

    max_grad_norm: float | None = None

    def train_one_epoch(self, dataloader: DataLoader[tuple[Tensor, ...]], epoch: int) -> float:
        """Single training epoch with optional gradient-norm clipping. Mirrors
        `TorchTrainer.train_one_epoch` but inserts `torch.nn.utils.clip_grad_norm_` between the
        backward pass and the optimizer step (unscaling first when mixed precision is on)."""

        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch",
                    desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})")
        for batch in pbar:
            X_batch, y_batch = batch
            X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
            y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

            self.initialized_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.initialized_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.initialized_optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        torch.cuda.empty_cache()
        gc.collect()
        return sum(losses) / len(losses)

    def save_model_to_external(self) -> None:
        """Save the model to external storage."""

        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(self.get_model_path())
            wandb.log_artifact(model_artifact)

    def create_datasets(self, X, y, train_indices, val_indices):
        """Create the torch datasets for training and validation. `X` and `y` are unused — the
        dataset's clean trajectory paths are derived from `self.dataset.dataset_path`. Both
        sub-sample to a fraction of the full frame list per epoch so the training loop and
        per-epoch val loss are fast. The final eval (in `offline_training.py`) builds its own
        val_dataset with `epoch_size_divisor=1` for an exact coverage number."""

        train_dataset = deepcopy(self.dataset)
        train_dataset.random_sampling = True
        train_dataset.initialize(train_indices)

        validation_dataset = deepcopy(self.dataset)
        validation_dataset.initialize(val_indices)

        return train_dataset, validation_dataset

    def create_prediction_dataset(self, x):
        """Create the torch prediction datasets for inference. `x` is interpreted as the indices
        into the sorted listing of `self.dataset.dataset_path`."""

        prediction_dataset = deepcopy(self.dataset)
        prediction_dataset.initialize(x)

        return prediction_dataset

    def get_hash(self) -> str:
        """Get the hash of the main trainer."""

        if self._fold == -1:
            return self._hash
        return f"{self._hash}_f{self._fold}"

    def _load_model(self, path: Path | None = None) -> None:
        """Load the model from the model_directory folder."""

        model_path = path if path is not None else self.get_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found in {model_path}")

        self.log_to_terminal(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, weights_only=False)
        model = checkpoint.module if isinstance(checkpoint, nn.DataParallel) else checkpoint

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

"""File containing functions related to setting up Weights and Biases."""
from pathlib import Path
from copy import deepcopy
import wandb
from ipdb import set_trace
from omegaconf import DictConfig, OmegaConf
from src.utils.logger import logger


def setup_wandb(cfg: DictConfig, job_type: str, output_dir: Path):
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :param job_type: The type of job, e.g. Training, CV, etc.
    :param output_dir: The directory to the Hydra outputs."""

    logger.debug("Initializing Weights & Biases")
    config = OmegaConf.to_container(cfg, resolve=True)

    if job_type == "test_tracking":
        run = wandb.init(
            config=replace_list_with_dict(config),
            project="test-tracking",
            entity="gregoire-andre-c-dumont",
            job_type=job_type,
            dir=output_dir,
            reinit=True)

    if job_type == "create_dataset":
        run = wandb.init(
            config=replace_list_with_dict(config),
            project="offline-training",
            entity="gregoire-andre-c-dumont",
            job_type=job_type,
            dir=output_dir,
            reinit=True)
        
    if job_type == "offline_training":
        run = wandb.init(
            config=replace_list_with_dict(config),
            project="offline-training",
            entity="gregoire-andre-c-dumont",
            job_type=job_type,
            dir=output_dir,
            reinit=True)
        



    logger.info("Done initializing Weights & Biases")
    return run

def replace_list_with_dict(o: object) -> object:
    """Recursively replace lists with integer index dicts."""
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = replace_list_with_dict(v)
    elif isinstance(o, list):
        o = {i: replace_list_with_dict(v) for i, v in enumerate(o)}
    return o
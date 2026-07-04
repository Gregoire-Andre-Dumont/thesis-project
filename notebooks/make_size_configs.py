"""Generate a BACKBONE-SIZE sweep: ONE experiment per tracker variant, each spanning
tiny / small / base_plus / large.

  exp 10  sam_baseline   (muggled_sam; auto-detects size from checkpoint)
  exp 11  samara_fixed   (method: SAM + learned controller; + per-size calibrator trainer)
  exp 12  sam2long       (official SAM2 loader; needs a per-size model_config yaml)
  exp 13  samurai        ( "" )
  exp 14  samite         ( "" )

For the official variants (12-14) it also DERIVES the missing conf/SAM2/<variant>_hiera_<size>.yaml
from the existing *_hiera_large.yaml by substituting the Hiera architecture block. It never
overwrites an existing model_config, and warns if a substitution string isn't found.

CAVEAT: the derived configs get the right *architecture* for the size, but inherit the large
config's *variant hyperparameters* (kf_score_weight / num_pathway / uncertainty / ...), which are
tuned per size in the official releases. Verify those against the official repos before trusting
the tiny/small SOTA-baseline numbers.

Run: python notebooks/make_size_configs.py
"""

from pathlib import Path

SIZES = ["tiny", "small", "base_plus", "large"]

# large -> size substitutions for the Hiera image-encoder block (official SAM2 params)
HIERA = {
    "tiny": [
        ("embed_dim: 144", "embed_dim: 96"), ("num_heads: 2", "num_heads: 1"),
        ("stages: [2, 6, 36, 4]", "stages: [1, 2, 7, 2]"),
        ("global_att_blocks: [23, 33, 43]", "global_att_blocks: [5, 7, 9]"),
        ("window_spec: [8, 4, 16, 8]", "window_spec: [8, 4, 14, 7]"),
        ("backbone_channel_list: [1152, 576, 288, 144]", "backbone_channel_list: [768, 384, 192, 96]"),
    ],
    "small": [
        ("embed_dim: 144", "embed_dim: 96"), ("num_heads: 2", "num_heads: 1"),
        ("stages: [2, 6, 36, 4]", "stages: [1, 2, 11, 2]"),
        ("global_att_blocks: [23, 33, 43]", "global_att_blocks: [7, 10, 13]"),
        ("window_spec: [8, 4, 16, 8]", "window_spec: [8, 4, 14, 7]"),
        ("backbone_channel_list: [1152, 576, 288, 144]", "backbone_channel_list: [768, 384, 192, 96]"),
    ],
    "base_plus": [
        ("embed_dim: 144", "embed_dim: 112"),
        ("stages: [2, 6, 36, 4]", "stages: [2, 3, 16, 3]"),
        ("global_att_blocks: [23, 33, 43]", "global_att_blocks: [12, 16, 20]"),
        ("window_spec: [8, 4, 16, 8]", "window_spec: [8, 4, 14, 7]"),
        ("backbone_channel_list: [1152, 576, 288, 144]", "backbone_channel_list: [896, 448, 224, 112]"),
        ("window_pos_embed_bkg_spatial_size: [7, 7]", "window_pos_embed_bkg_spatial_size: [14, 14]"),
    ],
    "large": [],
}

SAM2_DIR = Path("conf/SAM2")


def _exp(exp_num, variant, size_writer, trainer_writer=None):
    trk = Path(f"conf/trackers/experiment_{exp_num}")
    for old in trk.glob("*.yaml"):
        old.unlink()
    trk.mkdir(parents=True, exist_ok=True)
    if trainer_writer:
        trn = Path(f"conf/offline_trainers/experiment_{exp_num}")
        for old in trn.glob("*.yaml"):
            old.unlink()
        trn.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        (trk / f"{variant}_{size}.yaml").write_text(size_writer(size))
        if trainer_writer:
            (Path(f"conf/offline_trainers/experiment_{exp_num}") /
             f"cnn_fixed_samara_{size}.yaml").write_text(trainer_writer(size))
    print(f"exp {exp_num:2d}  {variant:13} -> {SIZES}")


def track_experiment(size, name):
    return (f"\nexperiment:\n"
            f"  _target_: src.experiments.tracking_experiment.TrackingExperiment\n"
            f"  resolution: 1024\n"
            f"  experiment_name: {name}_{size}\n"
            f"  video_path: data/visualize/{name}_{size}\n")


# ---- muggled variants (auto-detect size) ----
def sam_baseline(size):
    return (f"tracker:\n"
            f"  _target_: src.modules.baselines.sam_baseline.SAMBaseline\n"
            f"  iou_threshold: 0.2\n"
            f"  model:\n"
            f"    _target_: src.modules.samara_hiera_model.SamaraHieraModel\n"
            f"    sam_model_path: tm/sam_{size}.pt\n"
            f"    crop_resize: 384\n"
            f"    pad_ratio: 1.0\n"
            f"  main_memory:\n"
            f"    _target_: src.modules.memories.main_memory.MainMemory\n"
            f"    max_memory_history: 6\n"
            f"    max_pointer_history: 15\n"
            + track_experiment(size, "sam_baseline"))


def memory_oracle(size):
    return (f"tracker:\n"
            f"  _target_: src.modules.baselines.memory_oracle.MemoryOracle\n"
            f"  iou_threshold: 0.30\n"
            f"  model:\n"
            f"    _target_: src.modules.samara_hiera_model.SamaraHieraModel\n"
            f"    sam_model_path: tm/sam_{size}.pt\n"
            f"    crop_resize: 384\n"
            f"    pad_ratio: 1.0\n"
            f"  main_memory:\n"
            f"    _target_: src.modules.memories.main_memory.MainMemory\n"
            f"    max_memory_history: 6\n"
            f"    max_pointer_history: 15\n"
            + track_experiment(size, "memory_oracle"))


def samara_fixed(size):
    return (f"tracker:\n"
            f"  _target_: src.modules.samara_fixed.SamaraFixed\n"
            f"  iou_threshold: 0.5\n"
            f"  pred_iou_threshold: -1\n"
            f"  trainer_config_path: conf/offline_trainers/experiment_2/cnn_fixed_samara_{size}.yaml\n"
            f"  model:\n"
            f"    _target_: src.modules.samara_hiera_model.SamaraHieraModel\n"
            f"    sam_model_path: tm/sam_{size}.pt\n"
            f"    crop_resize: 512\n"
            f"    pad_ratio: 0.25\n"
            f"    token_source: memory\n"
            f"  main_memory:\n"
            f"    _target_: src.modules.memories.main_memory.MainMemory\n"
            f"    max_memory_history: 6\n"
            f"    max_pointer_history: 15\n"
            + track_experiment(size, "samara_fixed"))


def samara_trainer(size):
    return (f"main_trainer:\n"
            f"  _target_: src.offline_training.main_trainer.MainTrainer\n"
            f"  model_name: size_sweep_cnn_fixed_samara_{size}\n"
            f"  model:\n"
            f"    _target_: src.offline_training.classifiers.cnn_fixed.CNNFixed\n"
            f"    n_channels: 64\n    cnn_dim: 64\n    mlp_hidden: 512\n    dropout: 0.15\n    channel: channels_1_2\n"
            f"  criterion:\n    _target_: src.offline_training.losses.bce_iou.BCEIouLoss\n"
            f"  dataset:\n"
            f"    _target_: src.offline_training.main_dataset.MainDataset\n"
            f"    dataset_path: data/memory_oracle/{size}/memory/padding_0.25   # <-- PER-SIZE features; adjust\n"
            f"    epoch_size_divisor: 2\n    iou_threshold: 0.2\n"
            f"  collate_fn:\n    _target_: hydra.utils.get_method\n    path: src.offline_training.main_dataset.collate_fn\n"
            f"  optimizer:\n    _target_: functools.partial\n    _args_:\n"
            f"      - _target_: hydra.utils.get_class\n        path: torch.optim.Adam\n    lr: 0.000005002\n"
            f"  epochs: 15\n  batch_size: 32\n  patience: 30\n  n_folds: 5\n  checkpointing_resume_if_exists: False\n"
            f"  dataloader_args:\n    num_workers: 0\n")


# ---- official variants (need a per-size model_config yaml) ----
def derive_model_config(variant, size):
    """Create conf/SAM2/<variant>_hiera_<size>.yaml from *_hiera_large.yaml if it doesn't exist."""
    dst = SAM2_DIR / f"{variant}_hiera_{size}.yaml"
    if dst.exists() or size == "large":
        return
    src = SAM2_DIR / f"{variant}_hiera_large.yaml"
    if not src.exists():
        print(f"    ! {src} missing — cannot derive {dst.name}")
        return
    text = src.read_text()
    for old, new in HIERA[size]:
        if old not in text:
            print(f"    ! {dst.name}: substitution not found: '{old}' (verify architecture!)")
        text = text.replace(old, new)
    dst.write_text(text)
    print(f"    derived {dst.name} (architecture only; verify variant hyperparams)")


def official(variant, extra=""):
    def writer(size):
        derive_model_config(variant, size)
        cls = {"sam2long": "sam2long.SAM2Long", "samurai": "samurai.Samurai",
               "samite": "samite.Samite"}[variant]
        return (f"tracker:\n"
                f"  _target_: src.modules.baselines.{cls}\n"
                f"  checkpoint: tm/sam_{size}.pt\n"
                f"  model_config: SAM2/{variant}_hiera_{size}.yaml\n"
                f"{extra}"
                + track_experiment(size, variant))
    return writer


def main():
    _exp(1, "sam_baseline", sam_baseline)
    _exp(2, "samara_fixed", samara_fixed, trainer_writer=samara_trainer)
    _exp(3, "sam2long", official("sam2long", "  num_pathway: 2\n  iou_thre: 0.1\n  uncertainty: 2\n"))
    _exp(4, "samurai", official("samurai"))
    _exp(5, "samite", official("samite"))
    _exp(6, "memory_oracle", memory_oracle)


if __name__ == "__main__":
    main()

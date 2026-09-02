from dataclasses import dataclass

from src.modules.baselines.memory_oracle import MemoryOracle


@dataclass
class MaskOracle(MemoryOracle):
    """A MemoryOracle that also picks the mask by the oracle: among SAM 2's proposals it keeps the one with the
    highest true box IoU vs the GT box (an upper bound on per-frame mask selection), instead of SAM 2's own IoU
    token. The memory-commit gate (true box IoU > `iou_threshold`, visible frames only) is identical to the
    MemoryOracle -- only mask selection differs. All behaviour lives in MemoryOracle; this just flips the default."""

    oracle_mask_selection: bool = True

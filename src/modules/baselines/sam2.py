from dataclasses import dataclass

from src.modules.baselines.sam_baseline import SAMBaseline


@dataclass
class Sam2(SAMBaseline):
    """STANDARD SAM 2: every frame goes into the memory bank, as the released tracker does.

    SAM 2 has no commit rule. Its answer to occlusion lives entirely in the object head, which blanks the
    mask and swaps the object pointer for a learned no-object one, while the spatial memory is stored
    regardless -- annotated as occluded rather than withheld. That is the behaviour this arm reproduces,
    and it is the control every other arm in claim_1 is measured against: SAMURAI, SAMITE and SAM2Long
    each add a read-time rule on top of it, and SAMARA replaces it with a learned write-time gate.

    `SAMBaseline` differs in one line -- it keeps SAM's own confidence as a gate (`object_score > 0.5 and
    iou > iou_threshold`) -- which makes it a stronger, non-standard control. claim_2 uses that one; this
    one is the published tracker.
    """

    def should_commit(self, object_scores, iou_scores, chosen_mask, frame):
        """Always commit. Stock SAM 2 writes a memory for every frame it tracks."""

        return True

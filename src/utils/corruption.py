"""Memory-bank corruption: commit a CLEAN nearby distractor instead of the target.

`nearest_distractor_boxes` gives, per frame from the first occlusion onward, the box of the nearest OTHER person
(the target's last visible position is carried forward while it is occluded). It is NON-STICKY -- the injected
identity may switch frame to frame -- which models a tracker that grabs whoever happens to be closest rather than
latching onto one wrong person.

A tracker with `corruption_p > 0` and these boxes set will, at each of its own commit points, with that
probability box-prompt the distractor and write THAT into the memory bank instead of its own prediction. The
injected entry is a clean, well-formed mask of a real person -- the corruption is one of identity, not quality.
"""
import numpy as np


def nearest_distractor_boxes(detection_data, clean_boxes):
    """Per-frame nearest-other-person box (normalized xywh), or None where corruption cannot apply: before the
    first occlusion, before the target has ever been seen, or where no other person is annotated.

    Every frame from the first occlusion onward is corruptible, OCCLUDED FRAMES INCLUDED -- while the target is
    hidden its last visible position is carried forward and the nearest other person is measured from there.
    The injection is not tied to any tracker's commit gate, so all arms face the same corruption events."""

    boxes, frame_indices = detection_data.bboxes_norm, detection_data.frame_indices
    occlusions, shape = detection_data.occlusions, detection_data.frames[0].shape
    height, width, n_frames = shape[0], shape[1], len(boxes)
    first_occlusion = int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else n_frames

    corruption_boxes, last_box = [], None
    for index in range(n_frames):
        if float(boxes[index][2]) > 0:
            last_box = boxes[index]
        people = clean_boxes.get(int(frame_indices[index]), [])
        if index < first_occlusion or last_box is None or not people:
            corruption_boxes.append(None)
            continue
        reference_x = last_box[0] + last_box[2] / 2.0
        reference_y = last_box[1] + last_box[3] / 2.0
        nearest = min(people, key=lambda b: np.hypot((b[0] + b[2] / 2.0 - reference_x) * width,
                                                     (b[1] + b[3] / 2.0 - reference_y) * height))
        corruption_boxes.append(np.asarray(nearest, np.float32))
    return corruption_boxes

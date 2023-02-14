
from typing import Tuple, Optional

import numpy as np


def box_iou_batch(boxes_true: np.ndarray, boxes_detection: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union of two sets of bounding boxes - `boxes_true` and `boxes_detection`. Both sets of
    boxes are expected to be in `(x_min, y_min, x_max, y_max)` format.
    Args:
        boxes_true: 2d `np.ndarray` representing ground-truth boxes. `shape = (N, 4)` where N is number of true objects.
        boxes_detection: 2d `np.ndarray` representing detection boxes. `shape = (M, 4)` where M is number of detected objects.
    Returns:
        iou: 2d `np.ndarray` representing pairwise IoU of boxes from `boxes_true` and `boxes_detection`. `shape = (N, M)` where N is number of true objects and M is number of detected objects.
    Example:
    ```
    >>> import numpy as np
    >>> from onemetric.cv.utils.iou import box_iou_batch
    >>> boxes_true = np.array([
    ...     [0., 0., 1., 1.],
    ...     [2., 2., 2.5, 2.5]
    ... ])
    >>> boxes_detection = np.array([
    ...     [0., 0., 1., 1.],
    ...     [2., 2., 2.5, 2.5]
    ... ])
    >>> iou = box_iou_batch(boxes_true=boxes_true, boxes_detection=boxes_detection)
    >>> iou
    ... np.array([
    ...     [1., 0.],
    ...     [0., 1.]
    ... ])
    ```
    """

    _validate_boxes_batch(boxes_batch=boxes_true)
    _validate_boxes_batch(boxes_batch=boxes_detection)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(boxes_true.T)
    area_detection = box_area(boxes_detection.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_detection[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_detection[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)


def _validate_boxes_batch(boxes_batch: np.ndarray):
    if type(boxes_batch) != np.ndarray or len(boxes_batch.shape) != 2 or boxes_batch.shape[1] != 4:
        raise ValueError(
            f"Bounding boxes batch must be defined as 2d np.array with (N, 4) shape, {boxes_batch} given"
        )

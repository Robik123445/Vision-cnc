"""Post-processing utilities for YOLO segmentation outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def ensure_binary_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Resize and convert any input mask into strict binary mask {0,1}."""

    h, w = image_size
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8)


def _union_masks(masks: Iterable[np.ndarray], image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Create union of masks for one class while preserving original frame size."""

    union = None
    for mask in masks:
        binary = ensure_binary_mask(mask, image_size)
        union = binary if union is None else np.logical_or(union, binary)
    if union is None:
        return None
    return union.astype(np.uint8)


def process_yolo_masks(
    masks: List[np.ndarray],
    class_ids: List[int],
    confidences: List[float],
    class_map: Dict[int, str],
    image_size: Tuple[int, int],
    min_conf_by_class: Dict[str, float],
) -> tuple[dict, Dict[str, float]]:
    """Apply confidence gating and class-specific aggregation rules."""

    grouped_masks: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_conf: Dict[str, List[float]] = defaultdict(list)

    for mask, class_id, conf in zip(masks, class_ids, confidences):
        cls_name = class_map.get(class_id)
        if cls_name is None:
            continue
        min_conf = min_conf_by_class.get(cls_name, 0.0)
        if conf < min_conf:
            continue
        grouped_masks[cls_name].append(mask)
        grouped_conf[cls_name].append(float(conf))

    out_masks = {"workpiece": None, "clamp": None, "hand": None, "tool": None}
    out_conf = {"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0}

    workpiece_masks = grouped_masks.get("workpiece", [])
    if workpiece_masks:
        areas = [int(np.count_nonzero(ensure_binary_mask(mask, image_size))) for mask in workpiece_masks]
        idx = int(np.argmax(areas))
        out_masks["workpiece"] = ensure_binary_mask(workpiece_masks[idx], image_size)
        out_conf["workpiece"] = grouped_conf["workpiece"][idx]

    for cls_name in ("clamp", "hand", "tool"):
        class_masks = grouped_masks.get(cls_name, [])
        if class_masks:
            out_masks[cls_name] = _union_masks(class_masks, image_size)
            out_conf[cls_name] = max(grouped_conf[cls_name])

    return out_masks, out_conf

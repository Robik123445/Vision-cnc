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
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _union_masks(masks: Iterable[np.ndarray], image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Create union of masks for one class while preserving original frame size."""

    union = None
    for mask in masks:
        m = ensure_binary_mask(mask, image_size)
        union = m if union is None else np.logical_or(union, m)
    if union is None:
        return None
    return union.astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component for robust workpiece selection."""

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _close_and_fill(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply morphology close to bridge gaps and fill small holes."""

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop tiny noisy connected components under min_area pixels."""

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == idx] = 1
    return out


def process_yolo_masks(
    masks: List[np.ndarray],
    class_ids: List[int],
    confidences: List[float],
    class_map: Dict[int, str],
    image_size: Tuple[int, int],
    min_conf_by_class: Dict[str, float],
    min_area_by_class: Dict[str, int] | None = None,
) -> tuple[dict, Dict[str, float]]:
    """Apply confidence gating and class-specific aggregation rules."""

    min_area_by_class = min_area_by_class or {"clamp": 25, "hand": 25, "tool": 25}
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

    out_masks = {"workpiece": None, "clamp": None, "hand": None, "tool": None, "safety_mask": None}
    out_conf = {"workpiece": 0.0, "clamp": 0.0, "hand": 0.0, "tool": 0.0}

    workpiece_masks = grouped_masks.get("workpiece", [])
    if workpiece_masks:
        areas = [int(np.count_nonzero(ensure_binary_mask(mask, image_size))) for mask in workpiece_masks]
        idx = int(np.argmax(areas))
        workpiece = ensure_binary_mask(workpiece_masks[idx], image_size)
        workpiece = _largest_component(workpiece)
        workpiece = _close_and_fill(workpiece)
        out_masks["workpiece"] = workpiece.astype(np.uint8)
        out_conf["workpiece"] = grouped_conf["workpiece"][idx]

    for cls_name in ("clamp", "hand", "tool"):
        class_masks = grouped_masks.get(cls_name, [])
        if class_masks:
            union = _union_masks(class_masks, image_size)
            cleaned = _remove_small_components(union, min_area_by_class.get(cls_name, 25))
            out_masks[cls_name] = cleaned if np.count_nonzero(cleaned) else None
            out_conf[cls_name] = max(grouped_conf[cls_name])

    clamp = out_masks.get("clamp")
    hand = out_masks.get("hand")
    if clamp is not None or hand is not None:
        c = clamp if clamp is not None else np.zeros(image_size, dtype=np.uint8)
        h = hand if hand is not None else np.zeros(image_size, dtype=np.uint8)
        out_masks["safety_mask"] = np.logical_or(c, h).astype(np.uint8)

    return out_masks, out_conf

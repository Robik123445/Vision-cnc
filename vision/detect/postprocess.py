"""Post-processing utilities for YOLO segmentation outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ensure_binary_mask(mask: Any, image_size: Tuple[int, int]) -> Any:
    """Resize and convert any input mask into strict binary mask {0,1}."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    h, w = image_size
    binary = mask.astype(np.float32) if hasattr(mask, "astype") else mask
    if binary.shape[:2] != (h, w):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return (binary > 0.5).astype(np.uint8)


def _union_masks(masks: Iterable[Any], image_size: Tuple[int, int]) -> Optional[Any]:
    """Create union of masks for one class while preserving original frame size."""

    import numpy as np  # type: ignore

    union = None
    for mask in masks:
        binary = ensure_binary_mask(mask, image_size)
        union = binary if union is None else np.logical_or(union, binary)

    if union is None:
        return None
    return union.astype(np.uint8)


def _largest_component(mask: Any) -> Any:
    """Keep only the largest connected component for robust workpiece selection."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n_labels <= 1:
        return mask.astype(np.uint8)

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _close_and_fill(mask: Any, kernel_size: int = 7) -> Any:
    """Apply morphology close to bridge gaps and fill small holes."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def _remove_small_components(mask: Any, min_area: int) -> Any:
    """Drop tiny noisy connected components under min_area pixels."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, n_labels):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == idx] = 1
    return out


def process_yolo_masks(
    masks: List[Any],
    class_ids: List[int],
    confidences: List[float],
    class_map: Dict[int, str],
    image_size: Tuple[int, int],
    min_conf_by_class: Dict[str, float],
    min_area_by_class: Optional[Dict[str, int]] = None,
) -> tuple[dict, Dict[str, float]]:
    """Apply confidence gating and class-specific aggregation rules."""

    import numpy as np  # type: ignore

    min_area = {"clamp": 1, "hand": 1, "tool": 1}
    if min_area_by_class:
        min_area.update({key: int(value) for key, value in min_area_by_class.items()})

    grouped_masks: Dict[str, List[Any]] = defaultdict(list)
    grouped_conf: Dict[str, List[float]] = defaultdict(list)

    for mask, class_id, conf in zip(masks, class_ids, confidences):
        cls_name = class_map.get(class_id)
        if cls_name is None:
            continue
        if float(conf) < float(min_conf_by_class.get(cls_name, 0.0)):
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
        if not class_masks:
            continue
        union = _union_masks(class_masks, image_size)
        if union is None:
            continue
        cleaned = _remove_small_components(union, min_area.get(cls_name, 25))
        out_masks[cls_name] = cleaned if int(np.count_nonzero(cleaned)) > 0 else None
        out_conf[cls_name] = max(grouped_conf[cls_name])

    clamp = out_masks["clamp"]
    hand = out_masks["hand"]
    if clamp is not None or hand is not None:
        clamp_mask = clamp if clamp is not None else np.zeros(image_size, dtype=np.uint8)
        hand_mask = hand if hand is not None else np.zeros(image_size, dtype=np.uint8)
        out_masks["safety_mask"] = np.logical_or(clamp_mask, hand_mask).astype(np.uint8)

    return out_masks, out_conf

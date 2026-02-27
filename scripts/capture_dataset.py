#!/usr/bin/env python3
"""Capture raw dataset frames from webcam for later annotation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2


def main() -> int:
    """Stream webcam and save every Nth frame with metadata."""

    parser = argparse.ArgumentParser(description="Capture raw vision dataset")
    parser.add_argument("--out", default="dataset/raw", help="Output root folder")
    parser.add_argument("--every_n_frames", type=int, default=10, help="Save every Nth frame")
    parser.add_argument("--note", default="", help="Dataset note")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--machine_state", default="unknown", help="Optional machine state text")
    args = parser.parse_args()

    day = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(args.out) / day
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open camera {args.camera}")

    idx = 0
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("capture_dataset", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            if idx % max(args.every_n_frames, 1) == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = out_dir / f"{ts}.jpg"
                meta_path = out_dir / f"{ts}.json"
                cv2.imwrite(str(img_path), frame)
                payload = {
                    "timestamp": ts,
                    "note": args.note,
                    "machine_state": args.machine_state,
                    "camera": args.camera,
                }
                meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                saved += 1
            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""On-hardware smoke test: find the Reachy camera, grab one frame, save it.

Run from the voice venv (where the face dependencies live — a T1 decision,
see progress/T1.md):

    voice/.venv/bin/python face/check_camera.py [--out frame.jpg]

Prerequisites when running against the real camera: the vendor daemon must
have released the camera (or not be running at all), and this user must be
in the 'video' group — see face/camera.py's docstring for both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from camera import Camera


def candidates() -> list[tuple[int, str]]:
    """(index, human name) for every /dev/video* node, name from sysfs."""
    found = []
    for node in sorted(Path("/dev").glob("video*")):
        index = int(node.name.removeprefix("video"))
        name_file = Path(f"/sys/class/video4linux/{node.name}/name")
        name = name_file.read_text().strip() if name_file.exists() else "?"
        found.append((index, name))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="check_camera_frame.jpg",
                    help="where to save the grabbed frame")
    ap.add_argument("--device", type=int, default=None,
                    help="V4L2 index to use (default: first that yields a frame)")
    args = ap.parse_args()

    cams = candidates()
    if not cams:
        print("no /dev/video* devices at all — is the robot's USB plugged in?")
        return 1
    print("device candidates:")
    for index, name in cams:
        print(f"  /dev/video{index}: {name}")

    indices = [args.device] if args.device is not None else [i for i, _ in cams]
    for index in indices:
        print(f"trying /dev/video{index} ...")
        try:
            with Camera(index) as cam:
                frame = next(cam.frames(), None)
        except RuntimeError as exc:
            print(f"  {exc}")
            continue
        if frame is None:
            print("  opened, but no frame came out (metadata node? busy?)")
            continue
        h, w = frame.shape[:2]
        cv2.imwrite(args.out, frame)
        print(f"  got a {w}x{h} frame (mean brightness "
              f"{frame.mean():.1f}) -> saved to {args.out}")
        return 0

    print("no candidate produced a frame")
    return 1


if __name__ == "__main__":
    sys.exit(main())

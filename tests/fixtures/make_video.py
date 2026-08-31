"""Generate the webcam-style fixture clip from a still fixture photo.

The suite needs a short video containing a known face (T1's file-source
camera tests, T2's identify-in-video check). We have no recorded webcam
footage with cleared licensing, so this synthesizes one: a slow pan/zoom
over a public-domain portrait with slight brightness jitter and sensor-ish
noise, 640x480 @ 15 fps, MJPG in an AVI container (encodes everywhere,
no system codecs needed).

The output is committed (tests/fixtures/video/sunita_clip.avi); rerun this
only if the clip needs to change:

    tests/.venv/bin/python tests/fixtures/make_video.py
"""

from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "faces" / "sunita_b.jpg"
OUT = HERE / "video" / "sunita_clip.avi"

WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 15, 6


def main() -> None:
    src = cv2.imread(str(SRC))
    if src is None:
        raise SystemExit(f"cannot read {SRC}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"MJPG"),
                             FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("VideoWriter failed to open (MJPG/AVI)")

    h, w = src.shape[:2]
    rng = np.random.default_rng(20260831)
    n_frames = FPS * SECONDS
    for i in range(n_frames):
        t = i / (n_frames - 1)
        zoom = 1.0 + 0.15 * t                       # slow push in
        cw, ch = int(w / zoom), int(h / zoom)
        # gentle sinusoidal pan around the center
        cx = w / 2 + 0.04 * w * np.sin(2 * np.pi * t)
        cy = h / 2 + 0.03 * h * np.sin(2 * np.pi * t * 0.7)
        x0 = int(np.clip(cx - cw / 2, 0, w - cw))
        y0 = int(np.clip(cy - ch / 2, 0, h - ch))
        crop = src[y0:y0 + ch, x0:x0 + cw]
        frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # webcam texture: exposure flicker + a little sensor noise
        gain = 1.0 + rng.normal(0, 0.015)
        noise = rng.normal(0, 2.0, frame.shape)
        frame = np.clip(frame.astype(np.float32) * gain + noise,
                        0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    print(f"wrote {OUT} ({n_frames} frames, {WIDTH}x{HEIGHT} @ {FPS} fps)")


if __name__ == "__main__":
    main()

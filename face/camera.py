"""Frames from the Reachy Mini camera (or a hardware-free substitute).

One class, three interchangeable sources, selected by the ``source``
argument:

* an int (or digit string) — a live V4L2 device index. The Reachy camera
  shows up as ``/dev/video0`` (with ``/dev/video1`` as its metadata node —
  never open that one for frames).
* a path to a video file — for tests and offline runs.
* a path to a directory of images — for enrollment-from-snapshots flows.

``Camera.frames()`` yields OpenCV BGR images at a capped rate (default
~2 fps, the spec's between-sessions watch cadence, section 4A):

* Live device: frames are grabbed continuously but yielded at most every
  ``1/fps`` seconds (sleep-paced), so the consumer sees fresh frames, not a
  backlog of stale buffered ones.
* Video file: the clip's own timeline is subsampled to the capped rate
  (e.g. a 6 s, 15 fps clip at 2 fps yields ~12 frames) with **no
  wall-clock sleeping** — tests stay fast and deterministic.
* Image directory: every image, sorted by name, no pacing — snapshots are
  discrete, a rate cap means nothing there.

Access prerequisites for the live path, both documented from experience:

1. **The vendor daemon owns the camera.** When ``reachy-mini-daemon`` is
   running, ask the robot to hand its media over before opening the
   device — the driver's ``set_media_released`` command, which the voice
   agent already issues via ``RobotLink.release_media(True)``. With no
   daemon running the device is simply free.
2. **The user needs ``video`` group membership** (``sudo usermod -aG video
   $USER`` + a fresh login shell, or ``sg video -c ...``). Without it the
   open fails outright — same class of trap as ``dialout``/``audio`` in
   CLAUDE.md.

Usage::

    with Camera(0) as cam:                # live Reachy camera
        for frame in cam.frames():
            ...
            if done:
                break

    for frame in Camera("tests/fixtures/video/sunita_clip.avi").frames():
        ...
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Camera:
    """Rate-capped frame source over a V4L2 device, video file, or image dir."""

    def __init__(self, source: int | str | Path, fps: float = 2.0):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.closed = False
        self._cap: cv2.VideoCapture | None = None

        if isinstance(source, int) or (isinstance(source, str)
                                       and source.isdigit()):
            self._kind = "device"
            self._device_index = int(source)
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"camera source does not exist: {path}")
            if path.is_dir():
                self._kind = "images"
                self._images = sorted(
                    p for p in path.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES)
                if not self._images:
                    raise FileNotFoundError(f"no images in directory: {path}")
            else:
                self._kind = "video"
                self._video_path = path

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.closed = True

    # -- frames ------------------------------------------------------------

    def frames(self):
        """Yield BGR frames until the source ends or the consumer breaks.

        Always releases the underlying capture on the way out (normal end,
        ``break``, or an exception in the consumer), so a half-consumed
        generator never leaves the camera device held open.
        """
        try:
            if self._kind == "device":
                yield from self._device_frames()
            elif self._kind == "video":
                yield from self._video_frames()
            else:
                yield from self._image_frames()
        finally:
            self.close()

    def _open(self, target) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open {target!r} — for a live device, check that "
                "the vendor daemon has released the camera and that this "
                "user is in the 'video' group (see module docstring)")
        self._cap = cap
        return cap

    def _device_frames(self):
        cap = self._open(self._device_index)
        interval = 1.0 / self.fps
        next_due = time.monotonic()
        while True:
            # Drain the driver's buffer so the frame we decode is current,
            # then sleep out the rest of the interval.
            ok, frame = cap.read()
            if not ok:
                return
            now = time.monotonic()
            if now < next_due:
                time.sleep(min(next_due - now, interval))
                continue
            next_due = max(next_due + interval, now)
            yield frame

    def _video_frames(self):
        cap = self._open(str(self._video_path))
        native = cap.get(cv2.CAP_PROP_FPS) or 0
        # Sample the clip's timeline at the capped rate; a clip slower than
        # the cap just plays every frame.
        step = max(1, round(native / self.fps)) if native > 0 else 1
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if index % step == 0:
                yield frame
            index += 1

    def _image_frames(self):
        for path in self._images:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield frame

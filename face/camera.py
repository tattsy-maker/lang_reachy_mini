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

import threading
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

    @property
    def is_live(self) -> bool:
        """True for a V4L2 device (paced by the hardware); False for a file
        or image directory (paced by the consumer)."""
        return self._kind == "device"

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


class FrameHub:
    """One camera, many readers (T13.3).

    A V4L2 device can only be streamed by one ``VideoCapture`` at a time,
    and three things want the booth camera at once: the session watcher
    (presence + identity), the face tracker (position, every frame), and
    enrollment (a few snapshots mid-conversation). The hub owns the single
    ``Camera``, pumps it on a thread, and hands every reader the newest
    frame it has not seen yet -- a slow reader skips frames rather than
    lagging behind a backlog.

    File and directory sources are paced at the hub's ``fps`` on the wall
    clock, so a fixture clip behaves like a live camera to its readers
    (the T1 ``Camera`` deliberately does not sleep for files; the hub does,
    once, for everyone). When a file source ends, ``exhausted`` goes True
    and every ``frames()`` generator returns.

        hub = FrameHub(0, fps=2.0).start()
        for frame in hub.frames():         # any number of these
            ...
        hub.close()
    """

    def __init__(self, source, fps: float = 2.0):
        self.fps = fps
        self._camera = Camera(source, fps=fps)
        self._cond = threading.Condition()
        self._seq = 0
        self._frame = None
        self._closed = False
        self.exhausted = False
        self._thread: threading.Thread | None = None

    @property
    def is_live(self) -> bool:
        return self._camera.is_live

    def start(self) -> "FrameHub":
        if self._thread is None:
            self._thread = threading.Thread(target=self._pump, daemon=True,
                                            name="frame-hub")
            self._thread.start()
        return self

    def _pump(self) -> None:
        interval = 1.0 / self.fps
        try:
            for frame in self._camera.frames():
                if self._closed:
                    break
                with self._cond:
                    self._seq += 1
                    self._frame = frame
                    self._cond.notify_all()
                if not self._camera.is_live:
                    time.sleep(interval)
        finally:
            with self._cond:
                self.exhausted = True
                self._cond.notify_all()

    def latest(self):
        """(seq, frame) of the newest frame; frame is None before the first."""
        with self._cond:
            return self._seq, self._frame

    def next_after(self, seq: int, timeout: float | None = None):
        """Block until a frame newer than ``seq`` exists. Returns
        (seq, frame); frame is None on timeout or when the source is done."""
        with self._cond:
            deadline = None if timeout is None else time.monotonic() + timeout
            while self._seq <= seq and not self.exhausted:
                remaining = (None if deadline is None
                             else max(0.0, deadline - time.monotonic()))
                if remaining == 0.0:
                    return self._seq, None
                self._cond.wait(remaining)
            if self._seq <= seq:
                return self._seq, None
            return self._seq, self._frame

    def frames(self, timeout: float | None = None):
        """Yield each new frame; returns when the source is exhausted (or,
        with a timeout, when no frame arrives in time)."""
        seq = 0
        while True:
            seq, frame = self.next_after(seq, timeout)
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        self._closed = True
        with self._cond:
            self._cond.notify_all()
        self._camera.close()

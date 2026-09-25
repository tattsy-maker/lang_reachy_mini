"""T1: the Camera frame source — file paths fully, real hardware if present."""

import os

import pytest

from face.camera import Camera


@pytest.fixture
def clip(paths):
    return paths.fixtures / "video" / "sunita_clip.avi"


def test_video_frames_have_camera_shape(clip):
    frames = list(Camera(clip).frames())
    assert frames, "no frames from fixture clip"
    for frame in frames:
        assert frame.shape == (480, 640, 3)
        assert frame.dtype.name == "uint8"


def test_video_rate_cap_subsamples_timeline(clip):
    # 6 s, 15 fps, 90-frame clip: a 2 fps cap keeps every 8th frame -> 12;
    # a cap at (or above) native rate keeps everything.
    assert len(list(Camera(clip, fps=2.0).frames())) == 12
    assert len(list(Camera(clip, fps=15.0).frames())) == 90


def test_image_directory_source_yields_all_photos(paths):
    frames = list(Camera(paths.fixtures / "faces").frames())
    assert len(frames) == 6
    assert all(f.ndim == 3 for f in frames)


def test_breaking_out_releases_the_capture(clip):
    cam = Camera(clip)
    for _ in cam.frames():
        break
    assert cam.closed, "consumer broke out but the capture was not released"


def test_context_manager_closes(clip):
    with Camera(clip) as cam:
        next(cam.frames())
    assert cam.closed


def test_missing_source_raises():
    with pytest.raises(FileNotFoundError):
        Camera("no/such/file.avi")


def test_bad_fps_rejected(clip):
    with pytest.raises(ValueError):
        Camera(clip, fps=0)


@pytest.mark.robot
def test_real_reachy_camera_produces_a_nonblack_frame():
    if not os.access("/dev/video0", os.R_OK):
        pytest.skip("no read access to /dev/video0 "
                    "(needs 'video' group; see face/camera.py)")
    with Camera(0) as cam:
        frame = next(cam.frames(), None)
    assert frame is not None, "real camera opened but yielded no frame"
    assert frame.mean() > 5.0, "frame is essentially black"

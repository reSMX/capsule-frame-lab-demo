from __future__ import annotations

from web_demo.config import Settings


def test_file_limits_are_disabled_by_default(monkeypatch) -> None:
    for name in (
        "MAX_IMAGE_MB",
        "MAX_IMAGE_MEGAPIXELS",
        "MAX_VIDEO_MB",
        "MAX_VIDEO_SECONDS",
        "MAX_VIDEO_FRAMES",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.max_image_bytes is None
    assert settings.max_image_pixels is None
    assert settings.max_video_bytes is None
    assert settings.max_video_seconds is None
    assert settings.max_video_frames is None

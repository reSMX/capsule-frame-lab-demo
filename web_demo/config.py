"""Environment-backed configuration for the local-only demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path
    report_path: Path
    device: str
    max_image_bytes: int
    max_image_pixels: int
    max_video_bytes: int
    max_video_seconds: float
    video_sample_fps: float
    max_video_frames: int
    inference_batch_size: int
    max_preview_frames: int

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = Path(
            os.getenv("MODEL_PATH", str(ROOT_DIR / "runs" / "combined_galar_v1" / "best.pt"))
        ).expanduser()
        if not model_path.is_absolute():
            model_path = ROOT_DIR / model_path

        report_value = os.getenv("REPORT_PATH")
        report_path = Path(report_value).expanduser() if report_value else model_path.with_name("report.json")
        if not report_path.is_absolute():
            report_path = ROOT_DIR / report_path

        device = os.getenv("DEVICE", "auto").strip().lower()
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError("DEVICE must be one of: auto, cuda, cpu")

        return cls(
            model_path=model_path.resolve(),
            report_path=report_path.resolve(),
            device=device,
            max_image_bytes=_positive_int("MAX_IMAGE_MB", 15) * 1024 * 1024,
            max_image_pixels=_positive_int("MAX_IMAGE_MEGAPIXELS", 40) * 1_000_000,
            max_video_bytes=_positive_int("MAX_VIDEO_MB", 300) * 1024 * 1024,
            max_video_seconds=_positive_float("MAX_VIDEO_SECONDS", 120.0),
            video_sample_fps=_positive_float("VIDEO_SAMPLE_FPS", 1.0),
            max_video_frames=_positive_int("MAX_VIDEO_FRAMES", 120),
            inference_batch_size=_positive_int("INFERENCE_BATCH_SIZE", 16),
            max_preview_frames=min(_positive_int("MAX_PREVIEW_FRAMES", 12), 12),
        )

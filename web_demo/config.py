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


def _optional_limit_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() in {"", "0", "none", "unlimited", "off"}:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer or 'unlimited'") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero or 'unlimited'")
    return value


def _optional_limit_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() in {"", "0", "none", "unlimited", "off"}:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number or 'unlimited'") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero or 'unlimited'")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path
    report_path: Path
    device: str
    max_image_bytes: int | None
    max_image_pixels: int | None
    max_video_bytes: int | None
    max_video_seconds: float | None
    video_sample_fps: float
    max_video_frames: int | None
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
            max_image_bytes=(
                value * 1024 * 1024 if (value := _optional_limit_int("MAX_IMAGE_MB")) else None
            ),
            max_image_pixels=(
                value * 1_000_000
                if (value := _optional_limit_int("MAX_IMAGE_MEGAPIXELS"))
                else None
            ),
            max_video_bytes=(
                value * 1024 * 1024 if (value := _optional_limit_int("MAX_VIDEO_MB")) else None
            ),
            max_video_seconds=_optional_limit_float("MAX_VIDEO_SECONDS"),
            video_sample_fps=_positive_float("VIDEO_SAMPLE_FPS", 1.0),
            max_video_frames=_optional_limit_int("MAX_VIDEO_FRAMES"),
            inference_batch_size=_positive_int("INFERENCE_BATCH_SIZE", 16),
            max_preview_frames=min(_positive_int("MAX_PREVIEW_FRAMES", 12), 12),
        )

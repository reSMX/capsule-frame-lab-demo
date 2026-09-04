"""Strict media decoding and bounded video frame selection."""

from __future__ import annotations

import base64
import io
import math
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .inference import InferenceService
from .labels import describe_label


ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = None


class MediaValidationError(ValueError):
    """A user-provided media file failed a bounded validation step."""


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    if not data:
        raise MediaValidationError("Файл пуст. Выберите изображение JPEG, PNG или WebP.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            width, height = probe.size
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise MediaValidationError("Поддерживаются только изображения JPEG, PNG и WebP.")
            if width <= 0 or height <= 0:
                raise MediaValidationError("Изображение имеет некорректный размер.")
            if max_pixels > 0 and width * height > max_pixels:
                raise MediaValidationError("Изображение превышает допустимое число пикселей.")
            probe.verify()
        with Image.open(io.BytesIO(data)) as source:
            normalized = ImageOps.exif_transpose(source)
            normalized.load()
            return normalized.convert("RGB")
    except MediaValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise MediaValidationError("Не удалось прочитать изображение: файл повреждён или имеет неверный формат.") from exc


def validate_mp4_header(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(64)
    if not header:
        raise MediaValidationError("Видео-файл пуст.")
    if b"ftyp" not in header[:32]:
        raise MediaValidationError("Поддерживается только корректный контейнер MP4.")


def _frame_preview(image: Image.Image) -> str:
    preview = image.copy()
    preview.thumbnail((360, 240))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_video(path: Path, service: InferenceService, settings: Settings) -> dict[str, Any]:
    """Decode uniformly sampled frames, classify them in batches, and return bounded previews."""
    validate_mp4_header(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise MediaValidationError("Не удалось открыть MP4: возможно, кодек не поддерживается.")

    started = time.perf_counter()
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise MediaValidationError("Не удалось определить длительность видео.")
        duration = frame_count / fps
        if not math.isfinite(duration) or duration <= 0:
            raise MediaValidationError("Видео не содержит доступных кадров.")
        if settings.max_video_seconds > 0 and duration > settings.max_video_seconds:
            raise MediaValidationError(
                f"Видео длиннее допустимых {settings.max_video_seconds:g} секунд."
            )

        wanted = max(1, math.ceil(duration * settings.video_sample_fps))
        sample_count = min(wanted, frame_count)
        if settings.max_video_frames > 0:
            sample_count = min(sample_count, settings.max_video_frames)
        end_time = max(0.0, (frame_count - 1) / fps)
        if sample_count == 1:
            timestamps = [0.0]
        else:
            timestamps = [end_time * index / (sample_count - 1) for index in range(sample_count)]

        frames: list[Image.Image] = []
        actual_timestamps: list[float] = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            actual_timestamps.append(float(timestamp))
    finally:
        capture.release()

    if not frames:
        raise MediaValidationError("Не удалось декодировать кадры видео.")

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(frames), settings.inference_batch_size):
        predictions.extend(service.predict_images(frames[start : start + settings.inference_batch_size]))

    results = []
    for index, (timestamp, prediction) in enumerate(zip(actual_timestamps, predictions, strict=True)):
        results.append({"frame_index": index, "timestamp": round(timestamp, 3), **prediction})

    aggregate = {
        key: sum(item["all_scores"][key] for item in results) / len(results)
        for key in results[0]["all_scores"]
    }
    ranked_keys = sorted(aggregate, key=aggregate.__getitem__, reverse=True)
    summary_top = [
        {**describe_label(key), "score": float(aggregate[key])} for key in ranked_keys[:3]
    ]

    preview_indices = sorted(
        sorted(
            range(len(results)),
            key=lambda index: results[index]["top_prediction"]["score"],
            reverse=True,
        )[: settings.max_preview_frames]
    )
    previews = [
        {
            "frame_index": index,
            "timestamp": results[index]["timestamp"],
            "prediction": results[index]["top_prediction"],
            "image": _frame_preview(frames[index]),
        }
        for index in preview_indices
    ]
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "duration_seconds": float(round(duration, 3)),
        "sampled_frames": len(results),
        "frames": results,
        "summary_prediction": summary_top[0],
        "summary_top_predictions": summary_top,
        "previews": previews,
        "model": {"checkpoint_epoch": service.epoch, "device": service.device.type},
        "processing_ms": float(round(total_ms, 3)),
    }

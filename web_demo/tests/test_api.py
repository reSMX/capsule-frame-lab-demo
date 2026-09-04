from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import web_demo.app as app_module
from web_demo.app import create_app


def image_bytes(image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(120, 65, 55)).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image_format", "filename", "content_type"),
    [("PNG", "frame.png", "image/png"), ("JPEG", "frame.jpg", "image/jpeg")],
)
def test_valid_images_pass_api(settings, mock_service, image_format, filename, content_type) -> None:
    with TestClient(create_app(settings, mock_service)) as client:
        response = client.post(
            "/api/predict-image",
            files={"file": (filename, image_bytes(image_format), content_type)},
        )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["all_scores"]) == 14
    assert len(payload["top_predictions"]) == 3


@pytest.mark.parametrize(
    "payload",
    [b"not really an image", b""],
)
def test_invalid_or_empty_image_is_rejected(settings, mock_service, payload: bytes) -> None:
    with TestClient(create_app(settings, mock_service)) as client:
        response = client.post(
            "/api/predict-image",
            files={"file": ("frame.png", payload, "image/png")},
        )
    assert 400 <= response.status_code < 500


def test_oversized_image_is_rejected(settings, mock_service) -> None:
    constrained = settings.__class__(
        **{**{field: getattr(settings, field) for field in settings.__dataclass_fields__}, "max_image_bytes": 16}
    )
    with TestClient(create_app(constrained, mock_service)) as client:
        response = client.post(
            "/api/predict-image",
            files={"file": ("frame.png", image_bytes("PNG"), "image/png")},
        )
    assert response.status_code == 413


def test_health_and_model_schema(settings, mock_service) -> None:
    with TestClient(create_app(settings, mock_service)) as client:
        health = client.get("/api/health")
        model = client.get("/api/model")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "device": "cpu",
        "cuda_available": False,
        "checkpoint": "best.pt",
        "checkpoint_epoch": 3,
    }
    payload = model.json()
    assert payload["architecture"] == "EfficientNet-B0"
    assert payload["checkpoint_epoch"] == 3
    assert len(payload["classes"]) == 14
    assert len(payload["per_class_metrics"]) == 14


def build_test_mp4(path: Path) -> bytes:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 encoder is unavailable")
    for index in range(8):
        frame = np.full((48, 64, 3), (20 + index * 10, 70, 120), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_short_mp4_passes_and_temp_directory_is_removed(
    tmp_path: Path, settings, mock_service, monkeypatch
) -> None:
    source = tmp_path / "source.mp4"
    payload = build_test_mp4(source)
    request_temp = tmp_path / "request-temp"

    def fixed_temp_dir(*_args, **_kwargs) -> str:
        request_temp.mkdir()
        return str(request_temp)

    monkeypatch.setattr(app_module.tempfile, "mkdtemp", fixed_temp_dir)
    with TestClient(create_app(settings, mock_service)) as client:
        response = client.post(
            "/api/predict-video",
            files={"file": ("demo.mp4", payload, "video/mp4")},
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["sampled_frames"] >= 1
    assert len(result["frames"]) == result["sampled_frames"]
    assert len(result["previews"]) <= 12
    assert not request_temp.exists()


def test_bad_video_is_rejected_and_temp_directory_is_removed(
    tmp_path: Path, settings, mock_service, monkeypatch
) -> None:
    request_temp = tmp_path / "request-temp-bad"

    def fixed_temp_dir(*_args, **_kwargs) -> str:
        request_temp.mkdir()
        return str(request_temp)

    monkeypatch.setattr(app_module.tempfile, "mkdtemp", fixed_temp_dir)
    with TestClient(create_app(settings, mock_service)) as client:
        response = client.post(
            "/api/predict-video",
            files={"file": ("bad.mp4", b"not an mp4", "video/mp4")},
        )
    assert response.status_code == 422
    assert not request_temp.exists()

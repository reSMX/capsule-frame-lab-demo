from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from web_demo.config import Settings
from web_demo.labels import LABELS, describe_label


class MockInferenceService:
    epoch = 3
    device = SimpleNamespace(type="cpu")
    cuda_available = False

    def _result(self, offset: float = 0.0) -> dict[str, Any]:
        keys = list(LABELS)
        raw = [14 - index + offset for index in range(14)]
        total = sum(raw)
        scores = {key: float(value / total) for key, value in zip(keys, raw, strict=True)}
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        top = [{**describe_label(key), "score": scores[key]} for key in ranked[:3]]
        return {
            "top_prediction": top[0],
            "top_predictions": top,
            "all_scores": scores,
            "model": {"checkpoint_epoch": self.epoch, "device": "cpu"},
            "inference_ms": 4.2,
        }

    def predict_image(self, _image: Any) -> dict[str, Any]:
        return self._result()

    def predict_images(self, images: list[Any]) -> list[dict[str, Any]]:
        return [self._result(index / 100) for index, _ in enumerate(images)]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "device": "cpu",
            "cuda_available": False,
            "checkpoint": "best.pt",
            "checkpoint_epoch": 3,
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "architecture": "EfficientNet-B0",
            "checkpoint_epoch": 3,
            "validation_macro_f1": 0.29618118078579847,
            "evaluation_scope": "leak_free_video_level_validation",
            "overall_accuracy": 0.8096690136940733,
            "balanced_accuracy": 0.32048074673902666,
            "classes": [describe_label(key) for key in LABELS],
            "per_class_metrics": {
                key: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 1.0}
                for key in LABELS
            },
            "unsupported_classes": [],
        }


@pytest.fixture
def mock_service() -> MockInferenceService:
    return MockInferenceService()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_path=tmp_path / "best.pt",
        report_path=tmp_path / "report.json",
        device="cpu",
        max_image_bytes=2 * 1024 * 1024,
        max_image_pixels=2_000_000,
        max_video_bytes=5 * 1024 * 1024,
        max_video_seconds=10.0,
        video_sample_fps=1.0,
        max_video_frames=10,
        inference_batch_size=4,
        max_preview_frames=12,
    )

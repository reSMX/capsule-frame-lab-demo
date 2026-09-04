"""Safe, single-instance inference service for the trained EfficientNet-B0."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2

from .labels import describe_label


EXPECTED_CLASS_COUNT = 14


class ModelBusyError(RuntimeError):
    """Raised when the single inference slot is already occupied."""


class InferenceService:
    """Own one model instance and serialize access to it."""

    def __init__(self, checkpoint_path: Path, requested_device: str, report_path: Path | None = None):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise RuntimeError(f"Checkpoint not found at verified path: {self.checkpoint_path}")

        self.cuda_available = bool(torch.cuda.is_available())
        self.device = self._select_device(requested_device)
        self._lock = threading.Lock()

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load compatible checkpoint at verified path: {self.checkpoint_path}"
            ) from exc

        required = {"model", "classes", "epoch", "val_supported_macro_f1", "training_config"}
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise RuntimeError(
                f"Checkpoint at {self.checkpoint_path} is missing fields: {', '.join(missing)}"
            )

        self.class_to_index = self._validate_classes(checkpoint["classes"])
        self.index_to_class = [
            name for name, _ in sorted(self.class_to_index.items(), key=lambda item: item[1])
        ]
        self.epoch = int(checkpoint["epoch"])
        self.validation_macro_f1 = float(checkpoint["val_supported_macro_f1"])
        self.training_config = dict(checkpoint["training_config"])
        dropout = float(self.training_config.get("dropout", 0.2))
        if not 0 <= dropout < 1:
            raise RuntimeError(f"Invalid dropout value in checkpoint: {dropout}")

        self.model = efficientnet_b0(weights=None)
        self.model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, len(self.index_to_class)
        )
        try:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Model weights are incompatible with EfficientNet-B0 at: {self.checkpoint_path}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

        weights = EfficientNet_B0_Weights.DEFAULT
        reference = weights.transforms()
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(reference.mean, reference.std),
            ]
        )
        self.report = self._read_report(report_path)

    def _select_device(self, requested: str) -> torch.device:
        requested = requested.strip().lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if not self.cuda_available:
                raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available")
            return torch.device("cuda")
        if requested != "auto":
            raise RuntimeError("DEVICE must be one of: auto, cuda, cpu")
        return torch.device("cuda" if self.cuda_available else "cpu")

    @staticmethod
    def _validate_classes(raw: Any) -> dict[str, int]:
        if not isinstance(raw, dict) or not raw:
            raise RuntimeError("Checkpoint classes must be a non-empty mapping")
        parsed: dict[str, int] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("Checkpoint classes must map string keys to integer indices")
            parsed[key] = value
        indices = list(parsed.values())
        if len(set(indices)) != len(indices) or sorted(indices) != list(range(len(indices))):
            raise RuntimeError("Checkpoint class indices must be unique and contiguous from zero")
        if len(indices) != EXPECTED_CLASS_COUNT:
            raise RuntimeError(
                f"Checkpoint must contain {EXPECTED_CLASS_COUNT} classes, found {len(indices)}"
            )
        return parsed

    @staticmethod
    def _read_report(report_path: Path | None) -> dict[str, Any]:
        if report_path is None:
            return {}
        path = Path(report_path).resolve()
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read validation report at: {path}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Validation report must contain a JSON object: {path}")
        return value

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply exactly the deterministic validation transform."""
        rgb = image.convert("RGB")
        tensor = self.transform(rgb)
        if tuple(tensor.shape) != (3, 224, 224):
            raise RuntimeError(f"Unexpected preprocessed tensor shape: {tuple(tensor.shape)}")
        return tensor

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        return self.predict_images([image])[0]

    def predict_images(self, images: Iterable[Image.Image]) -> list[dict[str, Any]]:
        materialized = list(images)
        if not materialized:
            return []
        if not self._lock.acquire(blocking=False):
            raise ModelBusyError("Model is processing another request")
        started = time.perf_counter()
        try:
            batch = torch.stack([self.preprocess(image) for image in materialized]).to(self.device)
            with torch.inference_mode():
                logits = self._forward(batch)
                probabilities = torch.softmax(logits.float(), dim=1).cpu()
            total_ms = (time.perf_counter() - started) * 1000.0
        finally:
            self._lock.release()

        per_image_ms = total_ms / len(materialized)
        return [self._serialize_scores(row, per_image_ms) for row in probabilities]

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.device.type != "cuda":
            return self.model(batch)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(batch)
        except RuntimeError:
            torch.cuda.empty_cache()
            return self.model(batch)

    def _serialize_scores(self, row: torch.Tensor, inference_ms: float) -> dict[str, Any]:
        values = [float(value) for value in row.tolist()]
        if len(values) != EXPECTED_CLASS_COUNT or not all(math.isfinite(value) for value in values):
            raise RuntimeError("Model produced invalid class scores")
        ranked = sorted(range(len(values)), key=values.__getitem__, reverse=True)
        all_scores = {
            self.index_to_class[index]: float(values[index]) for index in range(len(values))
        }
        top_predictions = [
            {**describe_label(self.index_to_class[index]), "score": float(values[index])}
            for index in ranked[:3]
        ]
        return {
            "top_prediction": top_predictions[0],
            "top_predictions": top_predictions,
            "all_scores": all_scores,
            "model": {"checkpoint_epoch": self.epoch, "device": self.device.type},
            "inference_ms": float(round(inference_ms, 3)),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "device": self.device.type,
            "cuda_available": self.cuda_available,
            "checkpoint": self.checkpoint_path.name,
            "checkpoint_epoch": self.epoch,
        }

    def model_info(self) -> dict[str, Any]:
        report = self.report
        class_metrics = report.get("supported_classification_report", {})
        per_class = {
            key: value
            for key, value in class_metrics.items()
            if key in self.class_to_index and isinstance(value, dict)
        }
        return {
            "architecture": "EfficientNet-B0",
            "checkpoint_epoch": self.epoch,
            "validation_macro_f1": self.validation_macro_f1,
            "evaluation_scope": report.get("evaluation_scope"),
            "overall_accuracy": report.get("overall_accuracy"),
            "balanced_accuracy": report.get("supported_balanced_accuracy"),
            "classes": [describe_label(key) for key in self.index_to_class],
            "per_class_metrics": per_class,
            "unsupported_classes": list(report.get("unsupported_classes", [])),
        }

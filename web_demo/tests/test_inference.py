from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from web_demo.inference import InferenceService


def find_checkpoint() -> Path:
    candidates = []
    if os.getenv("MODEL_PATH"):
        candidates.append(Path(os.environ["MODEL_PATH"]))
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "runs" / "combined_galar_v1" / "best.pt",
            project_root.parent / "fgds" / "runs" / "combined_galar_v1" / "best.pt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    pytest.skip("The real combined_galar_v1 checkpoint is not available")


@pytest.fixture(scope="module")
def cpu_service() -> InferenceService:
    checkpoint = find_checkpoint()
    return InferenceService(checkpoint, "cpu", checkpoint.with_name("report.json"))


def test_preprocessing_shape_matches_validation(cpu_service: InferenceService) -> None:
    image = Image.new("RGB", (341, 267), color=(111, 72, 64))
    tensor = cpu_service.preprocess(image)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (3, 224, 224)


def test_real_checkpoint_cpu_returns_fourteen_finite_scores(cpu_service: InferenceService) -> None:
    image = Image.new("RGB", (256, 256), color=(91, 54, 47))
    result = cpu_service.predict_image(image)
    scores = list(result["all_scores"].values())
    assert len(scores) == 14
    assert all(isinstance(value, float) and math.isfinite(value) for value in scores)
    assert sum(scores) == pytest.approx(1.0, abs=1e-5)
    top_scores = [item["score"] for item in result["top_predictions"]]
    assert top_scores == sorted(top_scores, reverse=True)
    assert result["model"]["device"] == "cpu"
    assert result["model"]["checkpoint_epoch"] == 3


def test_missing_checkpoint_has_verified_path_in_error(tmp_path: Path) -> None:
    missing = (tmp_path / "missing.pt").resolve()
    with pytest.raises(RuntimeError, match="Checkpoint not found at verified path") as error:
        InferenceService(missing, "cpu")
    assert str(missing) in str(error.value)

"""Manual real-model smoke test for image and short MP4 inference."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from web_demo.config import Settings
from web_demo.inference import InferenceService
from web_demo.media import analyze_video


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (160, 120)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 encoder is unavailable")
    try:
        for index in range(8):
            x = np.linspace(15, 160, 160, dtype=np.uint8)
            channel = np.tile(x, (120, 1))
            frame = np.dstack(
                [
                    np.roll(channel, index * 4, axis=1),
                    np.full_like(channel, 58 + index * 3),
                    np.flip(channel, axis=1),
                ]
            )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    service = InferenceService(model_path, args.device, model_path.with_name("report.json"))
    image = Image.new("RGB", (320, 260), color=(104, 58, 50))
    warmup_result = service.predict_image(image)
    image_result = service.predict_image(image)

    settings = replace(
        Settings.from_env(),
        model_path=model_path,
        report_path=model_path.with_name("report.json"),
        device=args.device,
        max_video_seconds=10.0,
        video_sample_fps=1.0,
        max_video_frames=10,
        inference_batch_size=4,
    )
    with tempfile.TemporaryDirectory(prefix="capsule-smoke-") as directory:
        video_path = Path(directory) / "smoke.mp4"
        make_video(video_path)
        video_result = analyze_video(video_path, service, settings)

    print(
        json.dumps(
            {
                "device": service.device.type,
                "cuda_available": service.cuda_available,
                "checkpoint_epoch": service.epoch,
                "warmup_inference_ms": warmup_result["inference_ms"],
                "image_inference_ms": image_result["inference_ms"],
                "image_top_class": image_result["top_prediction"]["class_key"],
                "video_duration_seconds": video_result["duration_seconds"],
                "video_sampled_frames": video_result["sampled_frames"],
                "video_processing_ms": video_result["processing_ms"],
                "video_top_class": video_result["summary_prediction"]["class_key"],
                "temporary_directory_removed": True,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

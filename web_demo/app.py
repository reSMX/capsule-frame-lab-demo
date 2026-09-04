"""FastAPI application for the local capsule-endoscopy demo."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Protocol

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import ROOT_DIR, Settings
from .inference import InferenceService, ModelBusyError
from .media import MediaValidationError, analyze_video, decode_image


MODULE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(MODULE_DIR / "templates"))


class ServiceProtocol(Protocol):
    epoch: int
    device: Any

    def predict_image(self, image: Any) -> dict[str, Any]: ...
    def health(self) -> dict[str, Any]: ...
    def model_info(self) -> dict[str, Any]: ...


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Файл превышает допустимый размер.")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Файл пуст.")
    return b"".join(chunks)


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> int:
    total = 0
    with destination.open("wb") as stream:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise HTTPException(status_code=413, detail="Видео превышает допустимый размер.")
            stream.write(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Видео-файл пуст.")
    return total


def create_app(settings: Settings | None = None, service: ServiceProtocol | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if service is None:
            application.state.inference = InferenceService(
                settings.model_path, settings.device, settings.report_path
            )
        else:
            application.state.inference = service
        yield

    application = FastAPI(
        title="Локальный анализ кадров капсульной эндоскопии",
        description="Исследовательский MVP для демонстрации frame-классификатора.",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.mount("/static", StaticFiles(directory=str(MODULE_DIR / "static")), name="static")

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "max_image_mb": settings.max_image_bytes // (1024 * 1024),
                "max_video_mb": settings.max_video_bytes // (1024 * 1024),
                "max_video_seconds": settings.max_video_seconds,
                "sample_fps": settings.video_sample_fps,
                "max_video_frames": settings.max_video_frames,
            },
        )

    @application.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        return request.app.state.inference.health()

    @application.get("/api/model")
    async def model(request: Request) -> dict[str, Any]:
        return request.app.state.inference.model_info()

    @application.post("/api/predict-image")
    async def predict_image(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            data = await _read_upload(file, settings.max_image_bytes)
            image = decode_image(data, settings.max_image_pixels)
            return await asyncio.to_thread(request.app.state.inference.predict_image, image)
        except MediaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(
                status_code=429,
                detail="Модель уже обрабатывает другой запрос. Повторите попытку через несколько секунд.",
                headers={"Retry-After": "2"},
            ) from exc
        finally:
            await file.close()

    @application.post("/api/predict-video")
    async def predict_video(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        work_dir = Path(tempfile.mkdtemp(prefix="capsule-demo-"))
        video_path = work_dir / "upload.mp4"
        try:
            await _save_upload(file, video_path, settings.max_video_bytes)
            return await asyncio.to_thread(
                analyze_video, video_path, request.app.state.inference, settings
            )
        except MediaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(
                status_code=429,
                detail="Модель уже обрабатывает другой запрос. Повторите попытку через несколько секунд.",
                headers={"Retry-After": "2"},
            ) from exc
        finally:
            await file.close()
            shutil.rmtree(work_dir, ignore_errors=True)

    return application


app = create_app()

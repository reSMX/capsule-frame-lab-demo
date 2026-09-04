(() => {
  "use strict";

  const body = document.body;
  const config = {
    maxImageMb: Number(body.dataset.maxImageMb),
    maxVideoMb: Number(body.dataset.maxVideoMb),
    maxVideoSeconds: Number(body.dataset.maxVideoSeconds),
    sampleFps: Number(body.dataset.sampleFps),
  };
  const elements = Object.fromEntries([
    "runtimePill", "runtimeText", "imageTab", "videoTab", "uploadPanel", "fileInput",
    "dropZone", "dropTitle", "fileRules", "chooseButton", "filePreview", "previewMedia",
    "fileName", "fileMeta", "stateChip", "errorBox", "errorText", "analyzeButton",
    "analyzeButtonText", "resetButton", "statusText", "macroF1", "checkpointEpoch",
    "metricsBody", "accuracyNote", "resultSection", "resultDevice", "resultTime",
    "mainResultName", "mainResultCategory", "mainResultKey", "mainResultScore", "topBars",
    "allScores", "videoResults", "videoDuration", "videoFrameCount", "classFilter",
    "timelineBody", "galleryCard", "frameGallery"
  ].map((id) => [id, document.getElementById(id)]));

  let mode = "image";
  let selectedFile = null;
  let previewUrl = null;
  let modelInfo = null;
  let videoFrames = [];

  const percent = (score) => `${(Number(score) * 100).toFixed(1)}%`;
  const seconds = (value) => `${Number(value).toFixed(1)} с`;
  const formatBytes = (bytes) => bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} КБ`
    : `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;

  function labelFor(key) {
    const item = modelInfo?.classes?.find((entry) => entry.class_key === key);
    return item || { class_key: key, display_name: key, category: "класс модели" };
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setMode(nextMode) {
    if (nextMode === mode) return;
    mode = nextMode;
    reset(false);
    const isImage = mode === "image";
    elements.imageTab.classList.toggle("is-active", isImage);
    elements.videoTab.classList.toggle("is-active", !isImage);
    elements.imageTab.setAttribute("aria-selected", String(isImage));
    elements.videoTab.setAttribute("aria-selected", String(!isImage));
    elements.uploadPanel.setAttribute("aria-labelledby", isImage ? "imageTab" : "videoTab");
    elements.fileInput.accept = isImage ? "image/jpeg,image/png,image/webp" : "video/mp4";
    elements.dropTitle.textContent = isImage ? "Перетащите изображение сюда" : "Перетащите короткий MP4 сюда";
    elements.fileRules.textContent = isImage
      ? `JPEG, PNG или WebP · до ${config.maxImageMb} МБ · до 40 Мп`
      : `MP4 · до ${config.maxVideoMb} МБ · до ${config.maxVideoSeconds} с · ${config.sampleFps} кадр/с`;
  }

  function setError(message) {
    elements.errorText.textContent = message;
    elements.errorBox.hidden = false;
    elements.statusText.textContent = "Исправьте ошибку и попробуйте снова";
  }

  function clearError() {
    elements.errorBox.hidden = true;
    elements.errorText.textContent = "";
  }

  function validateSelection(file) {
    if (!file) return "Файл не выбран.";
    const maxBytes = (mode === "image" ? config.maxImageMb : config.maxVideoMb) * 1024 * 1024;
    if (file.size === 0) return "Файл пуст.";
    if (file.size > maxBytes) return `Файл больше допустимых ${mode === "image" ? config.maxImageMb : config.maxVideoMb} МБ.`;
    if (mode === "image" && !["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      return "Выберите JPEG, PNG или WebP.";
    }
    if (mode === "video" && file.type !== "video/mp4" && !file.name.toLowerCase().endsWith(".mp4")) {
      return "Выберите видео в контейнере MP4.";
    }
    return null;
  }

  function selectFile(file) {
    clearError();
    const error = validateSelection(file);
    if (error) {
      selectedFile = null;
      elements.analyzeButton.disabled = true;
      setError(error);
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    clearNode(elements.previewMedia);
    const media = document.createElement(mode === "image" ? "img" : "video");
    media.src = previewUrl;
    if (mode === "image") media.alt = "Локальное превью выбранного изображения";
    else {
      media.controls = true;
      media.muted = true;
      media.setAttribute("aria-label", "Локальное превью выбранного видео");
    }
    elements.previewMedia.appendChild(media);
    elements.fileName.textContent = file.name;
    elements.fileMeta.textContent = `${formatBytes(file.size)} · файл не отправляется во внешние сервисы`;
    elements.filePreview.hidden = false;
    elements.analyzeButton.disabled = false;
    elements.resetButton.hidden = false;
    elements.statusText.textContent = "Файл готов к анализу";
    elements.stateChip.innerHTML = "<i></i>Файл выбран";
    elements.resultSection.hidden = true;
  }

  function reset(clearMode = true) {
    selectedFile = null;
    elements.fileInput.value = "";
    elements.filePreview.hidden = true;
    elements.resultSection.hidden = true;
    elements.videoResults.hidden = true;
    elements.analyzeButton.disabled = true;
    elements.analyzeButton.classList.remove("is-loading");
    elements.analyzeButtonText.textContent = "Запустить анализ";
    elements.resetButton.hidden = true;
    elements.statusText.textContent = "Выберите файл, чтобы начать";
    clearError();
    videoFrames = [];
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    clearNode(elements.previewMedia);
    if (clearMode) elements.chooseButton.focus();
  }

  function makeScoreRow(prediction) {
    const row = document.createElement("div");
    row.className = "score-row";
    const head = document.createElement("div");
    head.className = "score-row-head";
    const name = document.createElement("strong");
    name.textContent = prediction.display_name;
    const value = document.createElement("span");
    value.textContent = percent(prediction.score);
    head.append(name, value);
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(0, Math.min(100, Number(prediction.score) * 100))}%`;
    track.appendChild(fill);
    const key = document.createElement("code");
    key.className = "score-key";
    key.textContent = prediction.class_key;
    row.append(head, track, key);
    return row;
  }

  function renderRanking(topPredictions, allScores) {
    clearNode(elements.topBars);
    topPredictions.forEach((prediction) => elements.topBars.appendChild(makeScoreRow(prediction)));
    clearNode(elements.allScores);
    Object.entries(allScores)
      .sort((a, b) => b[1] - a[1])
      .forEach(([key, score]) => {
        const item = document.createElement("div");
        item.className = "all-score-item";
        const name = document.createElement("span");
        name.textContent = labelFor(key).display_name;
        name.title = key;
        const value = document.createElement("span");
        value.textContent = percent(score);
        item.append(name, value);
        elements.allScores.appendChild(item);
      });
  }

  function renderPrimary(prediction, device, elapsedMs) {
    elements.mainResultName.textContent = prediction.display_name;
    elements.mainResultCategory.textContent = prediction.category;
    elements.mainResultKey.textContent = prediction.class_key;
    elements.mainResultScore.textContent = percent(prediction.score);
    elements.resultDevice.textContent = String(device).toUpperCase();
    elements.resultTime.textContent = `${Number(elapsedMs).toFixed(1)} мс`;
  }

  function renderImageResult(data) {
    renderPrimary(data.top_prediction, data.model.device, data.inference_ms);
    renderRanking(data.top_predictions, data.all_scores);
    elements.videoResults.hidden = true;
  }

  function renderTimeline(filter = "all") {
    clearNode(elements.timelineBody);
    const visible = videoFrames.filter((frame) => filter === "all" || frame.top_prediction.class_key === filter);
    visible.forEach((frame) => {
      const row = document.createElement("tr");
      [
        seconds(frame.timestamp),
        frame.top_prediction.display_name,
        frame.top_prediction.category,
        percent(frame.top_prediction.score),
      ].forEach((text) => {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.appendChild(cell);
      });
      elements.timelineBody.appendChild(row);
    });
    if (!visible.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 4;
      cell.textContent = "Для выбранного класса кадров нет.";
      row.appendChild(cell);
      elements.timelineBody.appendChild(row);
    }
  }

  function renderVideoResult(data) {
    const aggregateScores = {};
    data.frames.forEach((frame) => {
      Object.entries(frame.all_scores).forEach(([key, score]) => {
        aggregateScores[key] = (aggregateScores[key] || 0) + score / data.frames.length;
      });
    });
    renderPrimary(data.summary_prediction, data.model.device, data.processing_ms);
    renderRanking(data.summary_top_predictions, aggregateScores);
    elements.videoDuration.textContent = seconds(data.duration_seconds);
    elements.videoFrameCount.textContent = String(data.sampled_frames);
    videoFrames = data.frames;
    clearNode(elements.classFilter);
    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = "Все классы";
    elements.classFilter.appendChild(allOption);
    [...new Set(videoFrames.map((frame) => frame.top_prediction.class_key))]
      .sort((a, b) => labelFor(a).display_name.localeCompare(labelFor(b).display_name, "ru"))
      .forEach((key) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = labelFor(key).display_name;
        elements.classFilter.appendChild(option);
      });
    renderTimeline();
    clearNode(elements.frameGallery);
    data.previews.forEach((preview) => {
      const figure = document.createElement("figure");
      figure.className = "frame-card";
      const image = document.createElement("img");
      image.src = preview.image;
      image.alt = `Кадр на отметке ${seconds(preview.timestamp)}`;
      const caption = document.createElement("figcaption");
      const name = document.createElement("strong");
      name.textContent = preview.prediction.display_name;
      const meta = document.createElement("span");
      meta.textContent = `${seconds(preview.timestamp)} · ${percent(preview.prediction.score)}`;
      caption.append(name, meta);
      figure.append(image, caption);
      elements.frameGallery.appendChild(figure);
    });
    elements.galleryCard.hidden = data.previews.length === 0;
    elements.videoResults.hidden = false;
  }

  async function parseResponse(response) {
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error("Сервер вернул нечитаемый ответ."); }
    if (!response.ok) throw new Error(payload.detail || "Не удалось выполнить анализ.");
    return payload;
  }

  async function analyze() {
    if (!selectedFile) return;
    clearError();
    elements.analyzeButton.disabled = true;
    elements.analyzeButton.classList.add("is-loading");
    elements.analyzeButtonText.textContent = "Анализируем…";
    elements.statusText.textContent = mode === "image" ? "Модель анализирует кадр" : "Выбираем и анализируем кадры";
    elements.stateChip.innerHTML = "<i></i>Анализируется";
    const endpoint = mode === "image" ? "/api/predict-image" : "/api/predict-video";
    const form = new FormData();
    form.append("file", selectedFile);
    try {
      const response = await fetch(endpoint, { method: "POST", body: form });
      const data = await parseResponse(response);
      if (mode === "image") renderImageResult(data);
      else renderVideoResult(data);
      elements.resultSection.hidden = false;
      elements.statusText.textContent = "Результат готов";
      elements.stateChip.innerHTML = "<i></i>Результат готов";
      elements.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      setError(error instanceof Error ? error.message : "Неизвестная ошибка анализа.");
      elements.stateChip.innerHTML = "<i></i>Ошибка";
    } finally {
      elements.analyzeButton.disabled = false;
      elements.analyzeButton.classList.remove("is-loading");
      elements.analyzeButtonText.textContent = "Запустить снова";
    }
  }

  function renderModelInfo(info) {
    modelInfo = info;
    elements.macroF1.textContent = Number(info.validation_macro_f1).toFixed(3);
    elements.checkpointEpoch.textContent = String(info.checkpoint_epoch);
    clearNode(elements.metricsBody);
    info.classes.forEach((label) => {
      const metric = info.per_class_metrics[label.class_key] || {};
      const row = document.createElement("tr");
      const values = [
        label.display_name,
        Number(metric.precision || 0).toFixed(2),
        Number(metric.recall || 0).toFixed(2),
        Number(metric["f1-score"] || 0).toFixed(2),
        String(Number(metric.support || 0).toFixed(0)),
      ];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (index === 0) cell.title = label.class_key;
        row.appendChild(cell);
      });
      elements.metricsBody.appendChild(row);
    });
    elements.accuracyNote.textContent = info.overall_accuracy == null
      ? "Accuracy в отчёте недоступна."
      : `Общая accuracy: ${percent(info.overall_accuracy)}. Она зависит от дисбаланса классов и не является основной метрикой.`;
  }

  async function loadRuntime() {
    try {
      const [healthResponse, modelResponse] = await Promise.all([fetch("/api/health"), fetch("/api/model")]);
      const health = await parseResponse(healthResponse);
      const info = await parseResponse(modelResponse);
      elements.runtimePill.classList.add("is-ready");
      elements.runtimeText.textContent = `${health.device.toUpperCase()} · checkpoint ${health.checkpoint_epoch}`;
      renderModelInfo(info);
    } catch (error) {
      elements.runtimePill.classList.add("is-error");
      elements.runtimeText.textContent = "Модель недоступна";
      setError(error instanceof Error ? error.message : "Не удалось проверить модель.");
    }
  }

  elements.imageTab.addEventListener("click", () => setMode("image"));
  elements.videoTab.addEventListener("click", () => setMode("video"));
  elements.chooseButton.addEventListener("click", (event) => { event.stopPropagation(); elements.fileInput.click(); });
  elements.fileInput.addEventListener("change", () => selectFile(elements.fileInput.files[0]));
  ["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
    event.preventDefault(); elements.dropZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
    event.preventDefault(); elements.dropZone.classList.remove("is-dragging");
  }));
  elements.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
  elements.analyzeButton.addEventListener("click", analyze);
  elements.resetButton.addEventListener("click", () => reset(true));
  elements.classFilter.addEventListener("change", () => renderTimeline(elements.classFilter.value));

  loadRuntime();
})();

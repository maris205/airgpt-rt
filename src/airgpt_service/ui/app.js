const state = { models: [], current: null, mode: "vla", response: {} };

const $ = (id) => document.getElementById(id);
const requestStatus = $("requestStatus");
const responseOutput = $("responseOutput");
const preview = $("preview");

function setRequestStatus(label, kind = "") {
  requestStatus.textContent = label;
  requestStatus.className = `request-status ${kind}`.trim();
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("visible");
  window.setTimeout(() => node.classList.remove("visible"), 1800);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let body;
  try { body = await response.json(); } catch { body = { message: await response.text() }; }
  if (!response.ok) throw new Error(body.message || `Request failed (${response.status})`);
  return body;
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function currentModelName() {
  return $("modelSelect").value;
}

function displayModel(model) {
  state.current = model;
  if (!model) return;
  $("modelStatus").textContent = model.status;
  $("modelBackend").textContent = model.backend;
  $("modelDevice").textContent = model.device || "auto";
  $("modelId").textContent = model.metadata.model_id || "-";
  $("modelType").textContent = model.type.replace("_", " ");
  $("workspaceTitle").textContent = model.name;
  $("loadModel").disabled = model.status === "ready" || model.status === "loading";
  $("unloadModel").disabled = !model.loaded;
  document.querySelectorAll(".mode-tabs button").forEach((button) => {
    button.classList.toggle("hidden", button.dataset.mode !== model.type);
  });
  switchMode(model.type);
}

function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  document.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.panel !== mode);
  });
}

async function refreshModels(keepSelection = true) {
  try {
    const selected = keepSelection ? currentModelName() : "";
    const body = await api("/v1/models");
    state.models = body.models;
    const select = $("modelSelect");
    select.innerHTML = "";
    state.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = `${model.name} · ${model.status}`;
      select.appendChild(option);
    });
    if (selected && state.models.some((model) => model.name === selected)) select.value = selected;
    displayModel(state.models.find((model) => model.name === select.value) || state.models[0]);
    $("serviceState").className = "service-state online";
    $("serviceState").innerHTML = "<i></i> Online";
  } catch (error) {
    $("serviceState").className = "service-state offline";
    $("serviceState").innerHTML = "<i></i> Offline";
    setRequestStatus(error.message, "error");
  }
}

async function changeModelState(action) {
  const model = currentModelName();
  setRequestStatus(action === "load" ? "Loading" : "Unloading", "running");
  try {
    await api(`/v1/models/${model}/${action}`, { method: "POST" });
    await refreshModels();
    setRequestStatus(action === "load" ? "Ready" : "Unloaded", "success");
  } catch (error) {
    setRequestStatus(error.message, "error");
  }
}

function showResponse(body) {
  state.response = body;
  responseOutput.textContent = JSON.stringify(body, null, 2);
  preview.className = "preview";
  preview.innerHTML = "";
  if (body.action) return renderAction(body.action);
  if (body.frames) return renderWorld(body);
  if (body.latent_states) return renderState(body);
  preview.classList.add("empty-preview");
  preview.textContent = "Result preview";
}

function renderAction(action) {
  const labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"];
  const max = Math.max(...action.map((value) => Math.abs(value)), 0.000001);
  const chart = document.createElement("div");
  chart.className = "action-chart";
  action.forEach((value, index) => {
    const row = document.createElement("div");
    row.className = "action-row";
    row.innerHTML = `<span>${labels[index] || index}</span><div class="action-track"><div class="action-bar" style="width:${Math.max(2, Math.abs(value) / max * 100)}%"></div></div><span class="action-value">${Number(value).toFixed(6)}</span>`;
    chart.appendChild(row);
  });
  preview.appendChild(chart);
}

function renderWorld(body) {
  if (body.metadata?.video) {
    const video = document.createElement("video");
    video.controls = true;
    video.src = body.metadata.video;
    preview.appendChild(video);
  }
  const validFrames = body.frames.filter((url) => url.startsWith("/") || url.startsWith("http"));
  if (validFrames.length) {
    const grid = document.createElement("div");
    grid.className = "frame-grid";
    validFrames.slice(0, 9).forEach((url) => {
      const image = document.createElement("img");
      image.src = url;
      image.alt = "Generated frame";
      grid.appendChild(image);
    });
    preview.appendChild(grid);
  }
  if (!preview.children.length) {
    preview.classList.add("empty-preview");
    preview.textContent = `${body.frames.length} frames`;
  }
}

function renderState(body) {
  const metrics = [
    [body.latent_states.length, "Horizon"],
    [body.latent_states[0]?.length || 0, "Latent dim"],
    [body.predicted_observations?.[0]?.length || 0, "Observation dim"],
  ];
  const summary = document.createElement("div");
  summary.className = "state-summary";
  metrics.forEach(([value, label]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    metric.innerHTML = `<strong>${value}</strong><span>${label}</span>`;
    summary.appendChild(metric);
  });
  preview.appendChild(summary);
}

async function runInference(path, payload) {
  setRequestStatus("Running", "running");
  try {
    const body = await api(path, { method: "POST", body: JSON.stringify(payload) });
    showResponse(body);
    setRequestStatus("Complete", "success");
    await refreshModels();
  } catch (error) {
    showResponse({ error: error.message });
    setRequestStatus(error.message, "error");
  }
}

$("vlaForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("vlaImage").files[0];
  if (!file) return;
  const metadata = { seed: Number($("vlaSeed").value || 0) };
  if (state.current?.adapter === "octo") metadata.dataset = state.current.metadata.dataset;
  await runInference("/v1/robot/action", {
    model: currentModelName(), image: await fileToBase64(file),
    instruction: $("instruction").value, metadata,
  });
});

$("worldForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("worldImage").files[0];
  if (!file) return;
  await runInference("/v1/world/predict", {
    model: currentModelName(), image: await fileToBase64(file), action: $("worldAction").value,
    num_frames: Number($("frameCount").value), width: Number($("worldWidth").value),
    height: Number($("worldHeight").value), num_inference_steps: Number($("worldSteps").value),
    seed: 0,
  });
});

$("stateForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await runInference("/v1/state/predict", {
      model: currentModelName(), observation: JSON.parse($("observation").value),
      actions: JSON.parse($("actions").value), seed: Number($("stateSeed").value || 0), reset: true,
    });
  } catch (error) { setRequestStatus(`Invalid JSON: ${error.message}`, "error"); }
});

$("modelSelect").addEventListener("change", () => displayModel(state.models.find((model) => model.name === currentModelName())));
$("refreshModels").addEventListener("click", () => refreshModels());
$("loadModel").addEventListener("click", () => changeModelState("load"));
$("unloadModel").addEventListener("click", () => changeModelState("unload"));
$("copyResponse").addEventListener("click", async () => {
  await navigator.clipboard.writeText(JSON.stringify(state.response, null, 2));
  toast("Response copied");
});

refreshModels(false);

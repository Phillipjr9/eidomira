/**
 * Boost lab — runs the sub-pixel boost on the visitor's own device and reports what it cost.
 *
 * What this page is for: deciding whether the swap belongs in the browser. `app/boost.py`
 * proves the algorithm; this measures the *runtime* on real hardware, which is the part
 * nobody can answer from a server.
 *
 * What it deliberately does not do is pretend to be the product. There are no swap weights
 * on this server, so the "model" is a 0.3 KB stand-in that ORT genuinely executes — enough
 * to prove the runtime path end to end and to price the geometry. A real swapper is ~554 MB
 * and orders of magnitude slower, so every timing here is a floor.
 */
import { ALIGN, phases, sourceRectForPhase, interleave } from "../boost.js";

const ORT_VERSION = "1.30.0";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_MODULE = `${ORT_BASE}ort.webgpu.bundle.min.mjs`;
const MODEL_URL = "/static/lab/swapper-standin.onnx";
const AVERAGE_OVER = 30;
const PIXELS = ALIGN * ALIGN;

const $ = (id) => document.getElementById(id);

const state = {
  scale: 2,
  running: false,
  source: null,
  sourceKind: "none",
  stream: null,
  video: null,
  ort: null,
  session: null,
  provider: "—",
  region: null,
  frames: [],
  frame: 0,
};

// One reusable buffer per direction: allocating half a megabyte per pass on a phone is how a
// measurement becomes a memory test.
const packed = new Float32Array(3 * PIXELS);
const unpacked = new Float32Array(3 * PIXELS);
const rgba = new Uint8ClampedArray(4 * PIXELS);
const crop = document.createElement("canvas");
crop.width = ALIGN;
crop.height = ALIGN;
const cropContext = crop.getContext("2d", { willReadFrequently: true });

function log(message) {
  const line = `[${(performance.now() / 1000).toFixed(1)}s] ${message}`;
  const panel = $("log");
  panel.textContent += `${line}\n`;
  panel.scrollTop = panel.scrollHeight;
}

function readout(id, value, tone) {
  const node = $(id);
  if (node) {
    node.textContent = value;
    node.className = tone || "";
  }
}

// ─────────────────────────────── what this device is ───────────────────────────────

async function reportDevice() {
  const rows = [
    ["dWebgpu", "WebGPU", "navigator.gpu" in navigator ? "available" : "not available",
      "navigator.gpu" in navigator ? "good" : "warn"],
    ["dAdapter", "GPU adapter", "asking…", ""],
    ["dRuntime", "Runtime", `onnxruntime-web ${ORT_VERSION}`, ""],
    ["dProvider", "Provider in use", "not started", ""],
    ["dMemory", "Device memory",
      navigator.deviceMemory ? `${navigator.deviceMemory} GB (browser's estimate)` : "not reported",
      navigator.deviceMemory ? "" : "warn"],
    ["dThreads", "CPU threads", navigator.hardwareConcurrency ? `${navigator.hardwareConcurrency}` : "not reported", ""],
    ["dScreen", "Screen", `${window.innerWidth}×${window.innerHeight} @ ${window.devicePixelRatio}×` +
      (navigator.userAgentData ? ` · mobile: ${navigator.userAgentData.mobile}` : ""), ""],
    ["dHeap", "JS heap", performance.memory
      ? `${(performance.memory.usedJSHeapSize / 1048576).toFixed(1)} MB of ` +
        `${(performance.memory.jsHeapSizeLimit / 1048576).toFixed(0)} MB limit`
      : "not exposed by this browser", performance.memory ? "" : "warn"],
  ];
  $("device").innerHTML = rows
    .map(([id, label, value, tone]) => `<dt>${label}</dt><dd id="${id}" class="${tone}">${value}</dd>`)
    .join("");

  if (!("gpu" in navigator)) return;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      readout("dAdapter", "no adapter returned", "warn");
      return;
    }
    const info = adapter.info || {};
    const described = [info.vendor, info.architecture, info.description]
      .filter(Boolean).join(" · ") || "unnamed adapter";
    const limit = adapter.limits?.maxBufferSize
      ? ` · max buffer ${(adapter.limits.maxBufferSize / 1048576).toFixed(0)} MB` : "";
    readout("dAdapter", described + limit, "good");
  } catch (error) {
    readout("dAdapter", `request failed: ${error.message}`, "warn");
  }
}

// ───────────────────────────────── the runtime ─────────────────────────────────

async function loadRuntime() {
  if (state.session || state.ort === "failed") return;

  const started = performance.now();
  let ort;
  try {
    ort = await import(/* @vite-ignore */ ORT_MODULE);
  } catch (error) {
    state.ort = "failed";
    log(`could not load onnxruntime-web from the CDN: ${error.message}`);
    log("continuing in geometry-only mode: the boost runs, the inference does not.");
    readout("dProvider", "unavailable — geometry only", "warn");
    return;
  }

  ort.env.wasm.wasmPaths = ORT_BASE;
  // One thread on purpose: multi-threaded wasm needs SharedArrayBuffer, which needs
  // cross-origin isolation headers this application does not send. Asking for more threads
  // here would fail or fall back silently; saying so is better.
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = "error";
  state.ort = ort;
  log(`onnxruntime-web loaded in ${(performance.now() - started).toFixed(0)} ms ` +
      `(the WebGPU build is a ~28 MB wasm download the first time, cached after)`);

  const options = {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
  };
  const startedSession = performance.now();
  try {
    state.session = await ort.InferenceSession.create(MODEL_URL, options);
    state.provider = "webgpu";
  } catch (error) {
    log(`webgpu session failed (${error.message}); falling back to wasm`);
    try {
      state.session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"], graphOptimizationLevel: "all",
      });
      state.provider = "wasm (cpu)";
    } catch (fallbackError) {
      state.session = null;
      state.provider = "none — geometry only";
      log(`wasm session also failed: ${fallbackError.message}`);
    }
  }
  log(`session created in ${(performance.now() - startedSession).toFixed(0)} ms on ${state.provider}`);
  readout("dProvider", state.provider, state.provider.startsWith("webgpu") ? "good" : "warn");
  readout("dRuntime", `onnxruntime-web ${ORT_VERSION}`, "");
}

// ───────────────────────────────── the sources ─────────────────────────────────

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
  });
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();
  state.video = video;
  state.stream = stream;
  state.sourceKind = "camera";
  await new Promise((resolve) => {
    if (video.videoWidth) return resolve();
    video.onloadedmetadata = resolve;
  });
  // Only 1.5 sub-pixel steps of margin are needed; 6 px keeps every phase inside the frame
  // so no pass is sampled from an edge that does not exist.
  const side = Math.max(160, Math.min(320, Math.round(Math.min(video.videoWidth, video.videoHeight) * 0.5)));
  state.region = {
    x: Math.round((video.videoWidth - side) / 2),
    y: Math.round((video.videoHeight - side) / 2),
    w: side, h: side,
  };
  state.source = video;
  log(`camera ${video.videoWidth}×${video.videoHeight}, aligned region ${side}×${side} ` +
      `(so the 128 crop samples it at ${(side / ALIGN).toFixed(2)}× — below 1 is where the ` +
      `boost has detail to recover)`);
}

function startPattern() {
  const side = 256;
  const canvas = document.createElement("canvas");
  canvas.width = side;
  canvas.height = side;
  const context = canvas.getContext("2d");
  const image = context.createImageData(side, side);
  // Detail at and beyond the 128 grid's limit, plus large-scale structure so a scrambled
  // merge is obvious rather than merely slightly wrong.
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      const i = (y * side + x) * 4;
      const fine = (x + y) % 2 ? 40 : 0;
      const lines = (x % 7 === 0 ? 45 : 0) + (y % 11 === 0 ? 45 : 0);
      const wave = 55 + 45 * Math.sin((x / side) * Math.PI * 3) * Math.cos((y / side) * Math.PI * 2);
      const r = Math.max(0, Math.min(255, wave + fine + lines));
      image.data[i] = r;
      image.data[i + 1] = Math.max(0, Math.min(255, 200 - fine - lines + 20));
      image.data[i + 2] = Math.max(0, Math.min(255, 120 + fine));
      image.data[i + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  state.source = canvas;
  state.region = { x: 0, y: 0, w: side, h: side };
  state.sourceKind = "pattern";
  log(`test pattern ${side}×${side}, so the crop samples it at ${(side / ALIGN).toFixed(2)}×`);
}

// ───────────────────────────────── one frame ─────────────────────────────────

/** RGBA from the canvas into the NCHW float tensor the model wants. */
function pack(source) {
  const start = performance.now();
  for (let i = 0; i < PIXELS; i += 1) {
    const at = i * 4;
    packed[i] = source[at] / 255;
    packed[PIXELS + i] = source[at + 1] / 255;
    packed[2 * PIXELS + i] = source[at + 2] / 255;
  }
  return performance.now() - start;
}

/** The model's NCHW float output back to RGBA for display. */
function unpack(values) {
  const start = performance.now();
  for (let i = 0; i < PIXELS; i += 1) {
    const at = i * 4;
    rgba[at] = values[i] * 255;
    rgba[at + 1] = values[PIXELS + i] * 255;
    rgba[at + 2] = values[2 * PIXELS + i] * 255;
    rgba[at + 3] = 255;
  }
  return performance.now() - start;
}

/**
 * One pass: sample the crop at this phase, run the model, hand back its 128 square.
 *
 * The readback through `getImageData` is a real cost and a real design decision — a
 * production pipeline would keep the texture on the GPU instead. It is measured here rather
 * than hidden, because on a phone it is often the difference between usable and not.
 */
async function runPass(region, scale, phaseX, phaseY, ort) {
  const rect = sourceRectForPhase(region, scale, phaseX, phaseY);
  const sampleStart = performance.now();
  cropContext.clearRect(0, 0, ALIGN, ALIGN);
  cropContext.drawImage(state.source, rect.x, rect.y, rect.w, rect.h, 0, 0, ALIGN, ALIGN);
  const pixels = cropContext.getImageData(0, 0, ALIGN, ALIGN).data;
  const sampleMs = performance.now() - sampleStart;

  const packMs = pack(pixels);
  let inferMs = 0;
  let values;
  let unpackMs = 0;

  if (ort && state.session) {
    const inferStart = performance.now();
    // The same buffer every pass on purpose: `run` is awaited, so the model has consumed it
    // before the next pack overwrites it. Slicing here would allocate 200 KB nine times a
    // frame, which on a phone turns a measurement into a memory test.
    const tensor = new ort.Tensor("float32", packed, [1, 3, ALIGN, ALIGN]);
    const results = await state.session.run({ crop: tensor });
    inferMs = performance.now() - inferStart;
    // NCHW float, clipped: the stand-in is a convolution and a real one is not bounded either.
    const raw = results.face.data;
    for (let i = 0; i < unpacked.length; i += 1) {
      unpacked[i] = raw[i] < 0 ? 0 : raw[i] > 1 ? 1 : raw[i];
    }
    unpackMs = unpack(unpacked);
  } else {
    // Geometry-only mode: the identity stands in for the model, exactly as the Python tests
    // do, so the boost itself can still be measured and seen.
    for (let i = 0; i < PIXELS; i += 1) {
      const at = i * 4;
      rgba[at] = pixels[at];
      rgba[at + 1] = pixels[at + 1];
      rgba[at + 2] = pixels[at + 2];
      rgba[at + 3] = 255;
    }
  }
  return { data: Uint8ClampedArray.from(rgba), sampleMs, packMs, inferMs, unpackMs };
}

async function runFrame() {
  const { scale, region, ort } = state;
  const passes = [];
  const totals = { sample: 0, pack: 0, infer: 0, unpack: 0 };
  const frameStart = performance.now();

  for (const [phaseX, phaseY] of phases(scale)) {
    const pass = await runPass(region, scale, phaseX, phaseY, ort);
    totals.sample += pass.sampleMs;
    totals.pack += pass.packMs;
    totals.infer += pass.inferMs;
    totals.unpack += pass.unpackMs;
    passes.push({ x: phaseX, y: phaseY, data: pass.data, width: ALIGN, height: ALIGN, channels: 4 });
  }

  const mergeStart = performance.now();
  const view = $("view");
  if (scale === 1) {
    view.width = ALIGN;
    view.height = ALIGN;
    view.getContext("2d").putImageData(new ImageData(passes[0].data, ALIGN, ALIGN), 0, 0);
  } else {
    // `interleave` works in 3 channels, like the Python it mirrors; the display needs alpha,
    // so the RGB canvas is expanded into an RGBA frame here.
    const merged = interleave(
      passes.map((pass) => ({ ...pass, data: stripAlpha(pass.data), channels: 3 })), scale, 3,
    );
    const size = merged.width;
    const display = new Uint8ClampedArray(size * size * 4);
    for (let i = 0; i < size * size; i += 1) {
      display[i * 4] = merged.data[i * 3];
      display[i * 4 + 1] = merged.data[i * 3 + 1];
      display[i * 4 + 2] = merged.data[i * 3 + 2];
      display[i * 4 + 3] = 255;
    }
    view.width = size;
    view.height = size;
    view.getContext("2d").putImageData(new ImageData(display, size, size), 0, 0);
  }
  const mergeMs = performance.now() - mergeStart;

  state.frames.push({
    ...totals, merge: mergeMs, total: performance.now() - frameStart, passes: passes.length,
  });
  if (state.frames.length > AVERAGE_OVER) state.frames.shift();
  state.frame += 1;
  if (state.frame % 10 === 0) renderTimings();
}

function stripAlpha(source) {
  const out = new Uint8ClampedArray(PIXELS * 3);
  for (let i = 0; i < PIXELS; i += 1) {
    out[i * 3] = source[i * 4];
    out[i * 3 + 1] = source[i * 4 + 1];
    out[i * 3 + 2] = source[i * 4 + 2];
  }
  return out;
}

function renderTimings() {
  if (!state.frames.length) return;
  const average = (key) =>
    state.frames.reduce((sum, frame) => sum + frame[key], 0) / state.frames.length;
  const rows = [
    ["Crop sampling", average("sample")],
    ["Pack to tensor", average("pack")],
    ["Inference", average("infer")],
    ["Unpack", average("unpack")],
    ["Interleave + draw", average("merge")],
    ["Total per frame", average("total")],
  ];
  const passes = state.frames[state.frames.length - 1].passes;
  $("times").innerHTML =
    `<tr><th>${state.scale}× boost, ${passes} pass${passes > 1 ? "es" : ""} per frame</th><th>ms</th></tr>` +
    rows.map(([label, ms]) => `<tr><td>${label}</td><td>${ms.toFixed(2)}</td></tr>`).join("");
  const passLabel = $("passLabel");
  if (passLabel) passLabel.textContent = `· ${passes} pass${passes > 1 ? "es" : ""} at ${state.scale}×`;
}

// ───────────────────────────────── the controls ─────────────────────────────────

function stop() {
  state.running = false;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  state.video = null;
  $("start").hidden = false;
  $("stop").hidden = true;
  $("empty").hidden = false;
  $("empty").textContent = "Stopped. The camera has been released and nothing was recording.";
  renderTimings();
  log("stopped, camera released");
}

async function start(kind) {
  $("start").disabled = true;
  $("empty").hidden = true;
  try {
    await loadRuntime();
    if (kind === "pattern") startPattern(); else await startCamera();
  } catch (error) {
    log(`could not start: ${error.message}`);
    $("empty").hidden = false;
    $("empty").textContent = `Could not start: ${error.message}`;
    $("start").disabled = false;
    return;
  }
  state.running = true;
  state.frames = [];
  state.frame = 0;
  $("start").hidden = true;
  $("start").disabled = false;
  $("stop").hidden = false;
  log(`running at ${state.scale}× — ${phases(state.scale).length} passes per frame`);
  requestAnimationFrame(loop);
}

async function loop() {
  if (!state.running) return;
  try {
    await runFrame();
  } catch (error) {
    log(`frame failed: ${error.message}`);
    stop();
    return;
  }
  state.raf = requestAnimationFrame(loop);
}

function setScale(scale) {
  state.scale = scale;
  state.frames = [];
  document.querySelectorAll("[data-scale]").forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.scale) === scale));
  });
  const view = $("view");
  view.width = ALIGN * (scale === 1 ? 1 : scale);
  view.height = ALIGN * (scale === 1 ? 1 : scale);
  if (state.running) log(`switched to ${scale}× (${phases(scale).length} passes per frame)`);
  renderTimings();
}

document.addEventListener("DOMContentLoaded", () => {
  reportDevice();
  setScale(2);
  $("times").innerHTML = "<tr><td>Not running yet</td><td>—</td></tr>";
  $("start").addEventListener("click", () => start("camera"));
  $("pattern").addEventListener("click", () => start("pattern"));
  $("stop").addEventListener("click", stop);
  document.querySelectorAll("[data-scale]").forEach((button) => {
    button.addEventListener("click", () => setScale(Number(button.dataset.scale)));
  });
  // A measurement page that keeps running in a background tab measures the wrong thing and
  // drains the battery doing it.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.running) {
      log("tab hidden — pausing");
      stop();
    }
  });
});

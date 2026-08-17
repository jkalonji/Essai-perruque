import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Réglages de départ par perruque (ajustables en direct via les sliders).
// scale = largeur de la perruque exprimée en multiple de la largeur du visage détecté.
// vert / horiz = décalages exprimés en fraction de la largeur du visage.
// rot = rotation de base en degrés.
const DEFAULTS = {
  blonde_straight: { scale: 1.5, vert: 0.12, horiz: 0.0, rot: -4 },
  dark_wavy: { scale: 1.55, vert: 0.1, horiz: 0.0, rot: 3 },
  dark_straight: { scale: 1.45, vert: 0.1, horiz: -0.02, rot: 4 },
};

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("overlay"),
  stage: document.getElementById("stage"),
  hint: document.getElementById("hint"),
  wigPicker: document.getElementById("wigPicker"),
  snapBtn: document.getElementById("snapBtn"),
  mirrorBtn: document.getElementById("mirrorBtn"),
  scaleSlider: document.getElementById("scaleSlider"),
  vertSlider: document.getElementById("vertSlider"),
  horizSlider: document.getElementById("horizSlider"),
  rotSlider: document.getElementById("rotSlider"),
  resetBtn: document.getElementById("resetBtn"),
  gallerySection: document.getElementById("gallery-section"),
  gallery: document.getElementById("gallery"),
};

const ctx = els.canvas.getContext("2d");

let wigs = [];
let currentWig = null;
const wigImages = new Map(); // id -> HTMLImageElement
let mirrored = true;
let faceLandmarker = null;
let smoothed = null; // {x,y,angle,faceWidth}

function setHint(text) {
  els.hint.textContent = text;
  els.hint.classList.toggle("hidden", !text);
}

async function loadWigs() {
  const res = await fetch("assets/wigs.json");
  wigs = await res.json();
  wigs.forEach((w) => {
    const img = new Image();
    img.src = w.file;
    wigImages.set(w.id, img);
  });

  els.wigPicker.innerHTML = "";
  wigs.forEach((w) => {
    const btn = document.createElement("button");
    btn.className = "wig-option";
    btn.innerHTML = `<img src="${w.file}" alt="${w.name}" /><span>${w.name}</span>`;
    btn.addEventListener("click", () => selectWig(w.id));
    btn.dataset.id = w.id;
    els.wigPicker.appendChild(btn);
  });

  selectWig(wigs[0].id);
}

function selectWig(id) {
  currentWig = wigs.find((w) => w.id === id);
  [...els.wigPicker.children].forEach((btn) =>
    btn.classList.toggle("active", btn.dataset.id === id)
  );
  applyDefaults(id);
}

function applyDefaults(id) {
  const d = DEFAULTS[id] || { scale: 1.4, vert: 0.1, horiz: 0, rot: 0 };
  els.scaleSlider.value = d.scale;
  els.vertSlider.value = d.vert;
  els.horizSlider.value = d.horiz;
  els.rotSlider.value = d.rot;
}

els.resetBtn.addEventListener("click", () => {
  if (currentWig) applyDefaults(currentWig.id);
});

els.mirrorBtn.addEventListener("click", () => {
  mirrored = !mirrored;
  els.stage.classList.toggle("unmirrored", !mirrored);
});

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
  });
  els.video.srcObject = stream;
  await new Promise((resolve) => {
    els.video.onloadedmetadata = () => resolve();
  });
  els.video.play();
  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;
  els.snapBtn.disabled = false;
  els.mirrorBtn.disabled = false;
}

async function setupFaceLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}
function lerpAngle(a, b, t) {
  let diff = ((b - a + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
  return a + diff * t;
}

function computeTarget(landmarks) {
  const w = els.canvas.width;
  const h = els.canvas.height;
  const p = (i) => ({ x: landmarks[i].x * w, y: landmarks[i].y * h });

  const top = p(10); // haut du front / racine des cheveux
  const left = p(234); // tempe gauche
  const right = p(454); // tempe droite
  const eyeL = p(33);
  const eyeR = p(263);

  const faceWidth = Math.hypot(right.x - left.x, right.y - left.y);
  const angle = Math.atan2(eyeR.y - eyeL.y, eyeR.x - eyeL.x);

  return { x: top.x, y: top.y, angle, faceWidth };
}

function drawWig() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (!smoothed || !currentWig) return;
  const img = wigImages.get(currentWig.id);
  if (!img || !img.complete || img.naturalWidth === 0) return;

  const scale = parseFloat(els.scaleSlider.value);
  const vert = parseFloat(els.vertSlider.value);
  const horiz = parseFloat(els.horizSlider.value);
  const rotOffset = (parseFloat(els.rotSlider.value) * Math.PI) / 180;

  const s = (smoothed.faceWidth * scale) / currentWig.width;
  const drawW = currentWig.width * s;
  const drawH = currentWig.height * s;
  const anchorX = currentWig.anchorXNorm * drawW;
  const anchorY = currentWig.anchorYNorm * drawH;

  const offsetX = horiz * smoothed.faceWidth;
  const offsetY = -vert * smoothed.faceWidth;

  ctx.save();
  ctx.translate(smoothed.x + offsetX, smoothed.y + offsetY);
  ctx.rotate(smoothed.angle + rotOffset);
  ctx.drawImage(img, -anchorX, -anchorY, drawW, drawH);
  ctx.restore();
}

function renderLoop() {
  if (faceLandmarker && els.video.readyState >= 2) {
    const result = faceLandmarker.detectForVideo(els.video, performance.now());
    if (result.faceLandmarks && result.faceLandmarks.length > 0) {
      const target = computeTarget(result.faceLandmarks[0]);
      if (!smoothed) {
        smoothed = target;
      } else {
        smoothed = {
          x: lerp(smoothed.x, target.x, 0.4),
          y: lerp(smoothed.y, target.y, 0.4),
          angle: lerpAngle(smoothed.angle, target.angle, 0.3),
          faceWidth: lerp(smoothed.faceWidth, target.faceWidth, 0.3),
        };
      }
    } else {
      smoothed = null;
    }
  }
  drawWig();
  requestAnimationFrame(renderLoop);
}

function takeSnapshot() {
  const w = els.canvas.width;
  const h = els.canvas.height;
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const octx = out.getContext("2d");
  octx.save();
  if (mirrored) {
    octx.translate(w, 0);
    octx.scale(-1, 1);
  }
  octx.drawImage(els.video, 0, 0, w, h);
  octx.drawImage(els.canvas, 0, 0, w, h);
  octx.restore();

  const dataUrl = out.toDataURL("image/png");
  els.gallerySection.hidden = false;
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = `perruque-${currentWig ? currentWig.id : "photo"}-${Date.now()}.png`;
  const img = document.createElement("img");
  img.src = dataUrl;
  img.title = "Clique pour télécharger";
  img.style.cursor = "pointer";
  img.addEventListener("click", () => link.click());
  els.gallery.prepend(img);
}

els.snapBtn.addEventListener("click", takeSnapshot);

async function main() {
  try {
    await loadWigs();
  } catch (e) {
    console.error(e);
    setHint("Impossible de charger les perruques (assets/wigs.json).");
    return;
  }

  try {
    setHint("Chargement du suivi de visage…");
    await setupFaceLandmarker();
  } catch (e) {
    console.error(e);
    setHint(
      "Le suivi de visage n'a pas pu se charger (connexion internet nécessaire au premier lancement)."
    );
  }

  try {
    setHint("Autorise la caméra pour commencer…");
    await setupCamera();
    setHint("");
  } catch (e) {
    console.error(e);
    setHint("Caméra refusée ou indisponible. Autorise l'accès puis recharge la page.");
    return;
  }

  renderLoop();
}

main();

const els = {
  video: document.getElementById("video"),
  frozen: document.getElementById("frozen"),
  stage: document.getElementById("stage"),
  hint: document.getElementById("hint"),
  shootBtn: document.getElementById("shootBtn"),
  retakeBtn: document.getElementById("retakeBtn"),
  saveBtn: document.getElementById("saveBtn"),
  liveActions: document.getElementById("liveActions"),
  reviewActions: document.getElementById("reviewActions"),
  status: document.getElementById("status"),
};

const fctx = els.frozen.getContext("2d");

function setHint(text) {
  els.hint.textContent = text;
  els.hint.classList.toggle("hidden", !text);
}

function setStatus(text) {
  els.status.textContent = text;
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1600 }, height: { ideal: 1200 } },
    audio: false,
  });
  els.video.srcObject = stream;
  await new Promise((resolve) => {
    els.video.onloadedmetadata = () => resolve();
  });
  els.video.play();
  els.frozen.width = els.video.videoWidth;
  els.frozen.height = els.video.videoHeight;
  els.shootBtn.disabled = false;
}

function showLive() {
  els.video.classList.remove("hidden-canvas");
  els.frozen.classList.add("hidden-canvas");
  els.liveActions.hidden = false;
  els.reviewActions.hidden = true;
  setStatus("");
}

function showFrozen() {
  els.video.classList.add("hidden-canvas");
  els.frozen.classList.remove("hidden-canvas");
  els.liveActions.hidden = true;
  els.reviewActions.hidden = false;
}

els.shootBtn.addEventListener("click", () => {
  fctx.drawImage(els.video, 0, 0, els.frozen.width, els.frozen.height);
  showFrozen();
});

els.retakeBtn.addEventListener("click", showLive);

els.saveBtn.addEventListener("click", async () => {
  const blob = await new Promise((resolve) =>
    els.frozen.toBlob(resolve, "image/jpeg", 0.95)
  );

  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: "moi.jpg",
        types: [{ description: "Image JPEG", accept: { "image/jpeg": [".jpg"] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      setStatus("✅ Photo enregistrée. Tu peux revenir dans la conversation.");
      return;
    } catch (err) {
      if (err.name === "AbortError") {
        setStatus("Enregistrement annulé.");
        return;
      }
      console.error(err);
      // tombe sur le fallback ci-dessous en cas d'erreur inattendue
    }
  }

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "moi.jpg";
  a.click();
  URL.revokeObjectURL(url);
  setStatus(
    "✅ Photo téléchargée (dossier Téléchargements). Déplace-la dans le dossier images/ du projet, puis reviens dans la conversation."
  );
});

async function main() {
  try {
    setHint("Autorise la caméra pour commencer…");
    await setupCamera();
    setHint("");
  } catch (e) {
    console.error(e);
    setHint("Caméra refusée ou indisponible. Autorise l'accès puis recharge la page.");
  }
}

main();

// Cabine coiffure IA — logique cote client des 5 ecrans (specs/cabine-coiffure-ia.html
// section 05). La camera reprend la logique de webapp/capture.js, adaptee
// pour envoyer la photo au serveur (cabine_server.py) au lieu de
// l'enregistrer en local.

// Doit rester coherent avec ESTIMATED_SECONDS_PER_JOB dans cabine_server.py
// -> valeur de depart affichee avant le premier /api/status (qui donne
// ensuite la vraie valeur, decroissante, cf. syncCountdown plus bas).
const ESTIMATED_WAIT_SECONDS = 170;

const screens = {
  welcome: document.getElementById("screen-welcome"),
  photo: document.getElementById("screen-photo"),
  choice: document.getElementById("screen-choice"),
  wait: document.getElementById("screen-wait"),
  result: document.getElementById("screen-result"),
};

function showScreen(name) {
  for (const [key, el] of Object.entries(screens)) {
    el.hidden = key !== name;
  }
}

const state = {
  password: "",
  catalog: null, // { styles } depuis /api/styles
  photoBlob: null,
  style: null,
  jobId: null,
  afterUrl: null,
};

// --- écran 1 : accueil & mot de passe ---------------------------------

const els1 = {
  password: document.getElementById("password"),
  error: document.getElementById("welcomeError"),
  startBtn: document.getElementById("startBtn"),
};

els1.startBtn.addEventListener("click", async () => {
  const password = els1.password.value;
  els1.error.textContent = "";
  els1.startBtn.disabled = true;
  els1.startBtn.textContent = "Vérification…";
  try {
    const res = await fetch("/api/check_password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    const data = await res.json();
    if (!data.ok) {
      els1.error.textContent = "Mot de passe incorrect.";
      return;
    }
    state.password = password;
    await loadCatalog();
    showScreen("photo");
    setupCamera();
  } catch (e) {
    console.error(e);
    els1.error.textContent = "Impossible de joindre le serveur.";
  } finally {
    els1.startBtn.disabled = false;
    els1.startBtn.textContent = "Commencer →";
  }
});

async function loadCatalog() {
  if (state.catalog) return state.catalog;
  const res = await fetch("/api/styles");
  state.catalog = await res.json();
  return state.catalog;
}

// Lien profond depuis une fiche produit de la boutique (site/assets/site.js
// génère des liens "/essayer/?style=<id>") -> pré-sélectionne cette forme
// à l'arrivée sur l'écran de choix, sans empêcher d'en choisir une autre.
// Ne joue qu'une fois (mis à null après usage) pour ne pas re-forcer le
// choix si le testeur revient sur cet écran via "Changer de forme".
let preselectStyleId = new URLSearchParams(location.search).get("style");

// --- écran 2 : prise de photo (repris de webapp/capture.js) -----------

const els2 = {
  video: document.getElementById("video"),
  frozen: document.getElementById("frozen"),
  hint: document.getElementById("hint"),
  shootBtn: document.getElementById("shootBtn"),
  retakeBtn: document.getElementById("retakeBtn"),
  usePhotoBtn: document.getElementById("usePhotoBtn"),
  liveActions: document.getElementById("liveActions"),
  reviewActions: document.getElementById("reviewActions"),
};
const fctx = els2.frozen.getContext("2d");
let mediaStream = null;

function setHint(text) {
  els2.hint.textContent = text;
  els2.hint.classList.toggle("hidden", !text);
}

async function setupCamera() {
  try {
    setHint("Autorise la caméra pour commencer…");
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1600 }, height: { ideal: 1200 } },
      audio: false,
    });
    els2.video.srcObject = mediaStream;
    await new Promise((resolve) => (els2.video.onloadedmetadata = resolve));
    els2.video.play();
    els2.frozen.width = els2.video.videoWidth;
    els2.frozen.height = els2.video.videoHeight;
    els2.shootBtn.disabled = false;
    setHint("");
  } catch (e) {
    console.error(e);
    setHint("Caméra refusée ou indisponible. Autorise l'accès puis recharge la page.");
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

function showLive() {
  els2.video.classList.remove("hidden-canvas");
  els2.frozen.classList.add("hidden-canvas");
  els2.liveActions.hidden = false;
  els2.reviewActions.hidden = true;
}

function showFrozen() {
  els2.video.classList.add("hidden-canvas");
  els2.frozen.classList.remove("hidden-canvas");
  els2.liveActions.hidden = true;
  els2.reviewActions.hidden = false;
}

els2.shootBtn.addEventListener("click", () => {
  fctx.drawImage(els2.video, 0, 0, els2.frozen.width, els2.frozen.height);
  showFrozen();
});

els2.retakeBtn.addEventListener("click", showLive);

els2.usePhotoBtn.addEventListener("click", async () => {
  const blob = await new Promise((resolve) => els2.frozen.toBlob(resolve, "image/jpeg", 0.92));
  state.photoBlob = blob;
  stopCamera();
  showScreen("choice");
  renderChoiceScreen();
});

// --- écran 3 : choix de la forme ----------------------------------------
// Une seule couleur cote serveur (APP_COLOR dans cabine_server.py,
// appliquee automatiquement) -> plus d'etape de choix de couleur ici.

const els3 = {
  subtitle: document.getElementById("choiceSubtitle"),
  styleGrid: document.getElementById("styleGrid"),
  generateBtn: document.getElementById("generateBtn"),
};

function renderChoiceScreen() {
  state.style = null;
  els3.subtitle.textContent = "Quelle forme veux-tu essayer ?";
  els3.generateBtn.hidden = true;

  els3.styleGrid.innerHTML = "";
  let preselectBtn = null;
  for (const style of state.catalog.styles) {
    const btn = document.createElement("button");
    btn.className = "option-card";
    if (style.available) {
      btn.textContent = style.label;
      btn.addEventListener("click", () => selectStyle(style, btn));
      if (style.id === preselectStyleId) preselectBtn = { style, btn };
    } else {
      // Forme pas encore ouverte au public (cf. APP_STYLES dans
      // cabine_server.py) -> visible dans le catalogue mais desactivee,
      // plutot que retiree, pour montrer ce qui arrive.
      btn.classList.add("option-card--soon");
      btn.disabled = true;
      btn.innerHTML = `${style.label}<span class="option-card__soon">Bientôt disponible</span>`;
    }
    els3.styleGrid.appendChild(btn);
  }

  if (preselectBtn) {
    selectStyle(preselectBtn.style, preselectBtn.btn);
  }
  preselectStyleId = null;
}

function selectStyle(style, btn) {
  state.style = style;
  [...els3.styleGrid.children].forEach((c) => c.classList.remove("active"));
  btn.classList.add("active");
  els3.generateBtn.hidden = false;
}

els3.generateBtn.addEventListener("click", startGeneration);

// --- compte à rebours de l'écran d'attente -----------------------------
// Decompte cote client, resynchronise a chaque poll (toutes les 4s) sur
// l'eta_seconds renvoye par le serveur (qui decroit reellement pendant que
// le job tourne, cf. cabine_server.py) -> demande explicite : donner au
// testeur une idee du temps restant plutot qu'un spinner muet.
const waitTimerEl = document.getElementById("waitTimer");
let countdownInterval = null;
let countdownRemaining = 0;

function formatTimer(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  const rest = s % 60;
  return `${m}:${String(rest).padStart(2, "0")}`;
}

function syncCountdown(seconds) {
  countdownRemaining = seconds;
  waitTimerEl.textContent = formatTimer(countdownRemaining);
  if (countdownInterval) return;
  countdownInterval = setInterval(() => {
    countdownRemaining = Math.max(0, countdownRemaining - 1);
    waitTimerEl.textContent =
      countdownRemaining > 0 ? formatTimer(countdownRemaining) : "Presque prêt…";
  }, 1000);
}

function stopCountdown() {
  clearInterval(countdownInterval);
  countdownInterval = null;
}

async function startGeneration() {
  showScreen("wait");
  document.getElementById("waitMessage").textContent = "Envoi de la photo…";
  syncCountdown(ESTIMATED_WAIT_SECONDS);

  const form = new FormData();
  form.append("password", state.password);
  form.append("style", state.style.id);
  form.append("image", state.photoBlob, "photo.jpg");

  try {
    const res = await fetch("/api/generate", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "erreur serveur");
    state.jobId = data.job_id;
    pollStatus();
  } catch (e) {
    console.error(e);
    document.getElementById("waitMessage").textContent = "❌ " + e.message;
  }
}

async function pollStatus() {
  const waitMessage = document.getElementById("waitMessage");
  try {
    const res = await fetch(`/api/status/${state.jobId}`);
    const data = await res.json();

    if (data.status === "queued") {
      waitMessage.textContent =
        data.queue_position > 0
          ? `En file d'attente (${data.queue_position} devant toi)…`
          : "En file d'attente…";
      syncCountdown(data.eta_seconds);
    } else if (data.status === "running") {
      waitMessage.textContent = "Génération en cours…";
      syncCountdown(data.eta_seconds);
    } else if (data.status === "done") {
      stopCountdown();
      await showResult();
      return;
    } else if (data.status === "error") {
      stopCountdown();
      waitMessage.textContent = "❌ Échec de la génération : " + (data.error || "erreur inconnue");
      return;
    }
  } catch (e) {
    console.error(e);
    stopCountdown();
    waitMessage.textContent = "❌ Connexion perdue avec le serveur.";
    return;
  }
  setTimeout(pollStatus, 4000);
}

// --- écran 5 : résultat -------------------------------------------------

const els5 = {
  after: document.getElementById("resultAfter"),
  downloadBtn: document.getElementById("downloadBtn"),
  newStyleBtn: document.getElementById("newStyleBtn"),
};

async function showResult() {
  // Couleur deja appliquee cote serveur (APP_COLOR, cabine_server.py) avant
  // que le job ne passe "done" -> ce que /api/result renvoie est deja le
  // rendu final, plus besoin d'un second appel ici.
  setAfterImage(`/api/result/${state.jobId}/shape.png`);
  showScreen("result");
}

function setAfterImage(url) {
  els5.after.src = url;
  els5.downloadBtn.href = url;
}

els5.newStyleBtn.addEventListener("click", () => {
  showScreen("choice");
  renderChoiceScreen();
});

// L'attribut download d'un <a> est ignore par Safari iOS (il ouvre l'image
// dans un nouvel onglet au lieu de l'enregistrer) -> quand l'API Web Share
// avec fichiers est dispo (Safari iOS, Chrome Android), on passe par la
// feuille de partage native ("Enregistrer l'image" y fonctionne) ; sinon on
// laisse le comportement par defaut du lien (download, fiable sur desktop).
els5.downloadBtn.addEventListener("click", async (e) => {
  if (!(navigator.share && navigator.canShare)) return;
  e.preventDefault();
  try {
    const res = await fetch(els5.downloadBtn.href);
    const blob = await res.blob();
    const file = new File([blob], "cabine_coiffure.png", { type: blob.type || "image/png" });
    if (navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: "Ma coiffure IA" });
    } else {
      window.open(els5.downloadBtn.href, "_blank");
    }
  } catch (err) {
    console.error(err);
    window.open(els5.downloadBtn.href, "_blank");
  }
});

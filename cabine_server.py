"""
Serveur HTTP persistant du site "Meches" : sert le site vitrine/boutique de
demonstration (site/) sur "/" et "/boutique", et l'outil d'essai virtuel
("cabine coiffure IA", voir specs/cabine-coiffure-ia.html, cabine/) sous
l'onglet "/essayer/" -> garde le pipeline FLUX.1-Kontext-dev charge en
memoire pour toute la session au lieu de le recharger a chaque photo comme
le fait run_kontext.py en CLI.

Lancer (depuis la racine du depot, avec le venv dedie a FLUX) :
    ".venv/Scripts/python.exe" cabine_server.py
Puis ouvrir http://localhost:8080 en local pour tester, ou exposer via
cloudflared tunnel une fois valide (voir README, section "cabine coiffure").

Architecture (fig. 1 du spec) : une requete = une photo + une forme choisie.
La forme est mise en file (JobQueue plus bas) et traitee une a la fois par
un thread dedie qui garde le pipeline FLUX en memoire (le rechargement du
transformer FP8 depuis son cache local ne prend que quelques secondes, mais
recharger tout le pipeline a chaque requete comme run_kontext.py le fait en
CLI serait inutilement lent pour un usage serveur). Une seule couleur est
proposee (APP_COLOR, "noir") -> appliquee automatiquement juste apres la
forme, dans le meme job (post-traitement Lab quasi instantane, recolor_hair.py,
qui ne passe jamais par le GPU), plutot que de faire choisir une couleur au
testeur apres coup.

recolor_hair.py depend de cv2/mediapipe, installes dans l'environnement
Python SYSTEME (pas le .venv dedie a torch/diffusers/FLUX, cf. README) ->
on l'appelle en sous-processus avec l'executable Python systeme plutot que
de l'importer directement (evite un conflit de dependances entre les deux
environnements). Surcharger ce chemin via la variable d'environnement
CABINE_RECOLOR_PYTHON si "python" ne resout pas vers le bon interpreteur.
"""
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

import torch
from diffusers.utils import load_image
from flask import Flask, jsonify, request, send_from_directory, abort, redirect

import run_kontext as kontext

SITE_DIR = "site"
CABINE_DIR = "cabine"
SESSIONS_DIR = "server_sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

PASSWORD = os.environ.get("CABINE_PASSWORD", "coiffure2026")


def resolve_recolor_python():
    """Trouve un interpreteur Python qui a cv2+mediapipe pour recolor_hair.py.

    "python" ne resout pas forcement vers le bon interpreteur (selon le
    terminal, le PATH peut faire passer le .venv FLUX en premier, qui n'a
    pas cv2/mediapipe -> le sous-processus echouerait silencieusement cote
    UI, cf. bug remonte en session). On le verifie une fois au demarrage
    plutot que de decouvrir le probleme au premier essai couleur d'un
    testeur, et on log clairement l'interpreteur retenu.
    """
    override = os.environ.get("CABINE_RECOLOR_PYTHON")
    candidates = [override] if override else ["python", "python3", "py"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            proc = subprocess.run(
                [candidate, "-c", "import cv2, mediapipe"],
                capture_output=True, text=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0:
            print(f"-> couleur (recolor_hair.py) : interpreteur '{candidate}' OK (cv2+mediapipe trouves)")
            return candidate
        print(f"!! '{candidate}' resout vers un interpreteur SANS cv2/mediapipe -> essai suivant")
        print(proc.stderr.strip()[-500:])
    print(
        "!! ATTENTION: aucun interpreteur Python avec cv2+mediapipe trouve -> "
        "la couleur (post-traitement) va echouer sur tous les essais. "
        "Surcharge avec CABINE_RECOLOR_PYTHON=<chemin vers le python systeme>."
    )
    return override or "python"


RECOLOR_PYTHON = resolve_recolor_python()

# Sous-ensemble de HAIRSTYLES expose au testeur : les 5 "formes" de la
# matrice forme x couleur (spec section 07). curly_femme/
# brushing_frange_femme restent des styles d'exploration CLI (genre
# different de l'utilisateur, hors sujet pour une cabine d'essai) -> pas
# exposes ici.
APP_STYLES = [
    {"id": "brushing_frange", "label": "Brushing"},
    {"id": "attache", "label": "Attaché"},
    {"id": "frange", "label": "Frange"},
    {"id": "raie_milieu", "label": "Raie au milieu"},
    {"id": "curly", "label": "Bouclé"},
]
APP_STYLE_IDS = {s["id"] for s in APP_STYLES}

# Une seule couleur proposee au testeur (decision produit : plus la peine de
# choisir puisqu'il n'y a qu'une option) -> appliquee automatiquement en fin
# de generation, cf. worker_loop. "brun"/"blond" restaient trop sujets a
# derive d'identite quand testes directement dans le prompt FLUX (essai
# ecarte, voir historique) ; "noir" reste le post-traitement Lab existant
# (recolor_hair.py), qui ne touche jamais au visage. Choix entre brun_fonce
# et noir tranche par comparaison visuelle -> voir commentaire dans
# recolor_hair.py au-dessus de FULL_COLORS.
APP_COLOR = "noir"

# Estimation grossiere pour l'ecran d'attente (~2-4 min annonces dans le
# spec) -> une fois le pipeline chauffe, une generation tourne autour de
# 2h30-3min sur la RTX 5060 Ti (cf. logs de run_kontext.py).
ESTIMATED_SECONDS_PER_JOB = 170


class Job:
    def __init__(self, job_id, style, input_path):
        self.id = job_id
        self.style = style
        self.input_path = input_path
        self.status = "queued"  # queued | running | done | error
        self.shape_path = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None  # pose quand le job passe en "running" -> sert a calculer un ETA qui decroit reellement


JOBS = {}
JOBS_LOCK = threading.Lock()
QUEUE = deque()
QUEUE_COND = threading.Condition()


def apply_color(shape_path):
    """Lance recolor_hair.py (post-traitement Lab, APP_COLOR fixe) sur la
    forme fraichement generee par FLUX, et renvoie le chemin du resultat
    colore. Leve une exception si le sous-processus echoue -> le job passe
    en "error" (worker_loop), pas de resultat noir/silencieux.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [RECOLOR_PYTHON, "recolor_hair.py", shape_path, APP_COLOR],
        capture_output=True, text=True, cwd=repo_dir,
    )
    if proc.stdout.strip():
        print("-> recolor_hair.py:", proc.stdout.strip())
    if proc.returncode != 0:
        print(f"!! echec recolor_hair.py (interpreteur={RECOLOR_PYTHON!r}, code={proc.returncode}):")
        print(proc.stderr[-2000:])
        raise RuntimeError(f"echec du post-traitement couleur : {proc.stderr.strip()[-300:]}")

    out_path = os.path.join(os.path.dirname(shape_path), f"shape_{APP_COLOR}.jpg")
    if not os.path.exists(out_path):
        raise RuntimeError("fichier de sortie introuvable apres post-traitement couleur")
    return out_path


def enqueue(job):
    with JOBS_LOCK:
        JOBS[job.id] = job
    with QUEUE_COND:
        QUEUE.append(job.id)
        QUEUE_COND.notify()


def queue_position(job_id):
    """Nombre de jobs devant celui-ci (0 = le prochain traite, ou en cours)."""
    with QUEUE_COND:
        try:
            return list(QUEUE).index(job_id)
        except ValueError:
            return 0  # deja sorti de la file (en cours ou termine)


def worker_loop():
    # Pipeline charge une seule fois, au premier job -> demarrage du serveur
    # instantane, la generation (deja rapide grace au cache FP8) n'attend
    # que le premier testeur. current_lora suit quel LoRA est charge sur le
    # pipeline pour ne le recharger/decharger que quand le style change
    # (fig. 1 du spec : "swap LoRA selon le style choisi").
    pipe = None
    current_lora = None

    while True:
        with QUEUE_COND:
            while not QUEUE:
                QUEUE_COND.wait()
            job_id = QUEUE.popleft()

        with JOBS_LOCK:
            job = JOBS[job_id]

        try:
            job.status = "running"
            job.started_at = time.time()
            prompt, lora_repo = kontext.HAIRSTYLES[job.style]

            if pipe is None:
                print("-> chargement du pipeline FLUX (une seule fois pour toute la session)...")
                pipe = kontext.build_pipe(lora_repo=None)
                kontext.patch_attention(pipe)
                print("-> pipeline pret, reste en memoire pour les prochaines requetes")

            if lora_repo != current_lora:
                if current_lora is not None:
                    pipe.unload_lora_weights()
                if lora_repo is not None:
                    print(f"-> chargement du LoRA {lora_repo}...")
                    pipe.load_lora_weights(lora_repo)
                current_lora = lora_repo

            image = load_image(job.input_path)
            if max(image.size) > kontext.MAX_SIDE:
                ratio = kontext.MAX_SIDE / max(image.size)
                new_size = (round(image.width * ratio), round(image.height * ratio))
                image = image.resize(new_size)

            generator = torch.Generator(device="cuda").manual_seed(kontext.SEED)
            print(f"-> generation job={job.id} style={job.style}...")
            result = pipe(
                image=image, prompt=prompt, guidance_scale=2.5,
                num_inference_steps=kontext.NUM_STEPS, generator=generator,
            ).images[0]

            shape_path = os.path.join(SESSIONS_DIR, job.id, "shape.png")
            result.save(shape_path)

            job.shape_path = apply_color(shape_path)
            job.status = "done"
            print("-> termine:", job.shape_path)
        except Exception as e:
            job.error = str(e)
            job.status = "error"
            print(f"!! erreur job {job.id}: {e}")
        finally:
            with QUEUE_COND:
                QUEUE_COND.notify_all()


threading.Thread(target=worker_loop, daemon=True).start()

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 Mo, large marge pour une photo telephone


@app.route("/")
def index():
    return send_from_directory(SITE_DIR, "index.html")


@app.route("/boutique")
def boutique():
    return send_from_directory(SITE_DIR, "boutique.html")


@app.route("/assets/<path:filename>")
def site_assets(filename):
    return send_from_directory(os.path.join(SITE_DIR, "assets"), filename)


# Onglet "Essayer des meches" : l'outil FLUX existant (cabine/), monte sous
# /essayer/ plutot qu'a la racine. Slash final obligatoire pour que les
# chemins relatifs de cabine/index.html (app.js, style.css) se resolvent
# sous /essayer/ au lieu de la racine du site -> on redirige /essayer (sans
# slash) au cas ou un lien oublierait le slash.
@app.route("/essayer")
def essayer_redirect():
    return redirect("/essayer/")


@app.route("/essayer/")
def essayer_index():
    return send_from_directory(CABINE_DIR, "index.html")


@app.route("/essayer/<path:filename>")
def essayer_static(filename):
    return send_from_directory(CABINE_DIR, filename)


@app.route("/api/styles")
def api_styles():
    return jsonify(styles=APP_STYLES)


@app.route("/api/check_password", methods=["POST"])
def api_check_password():
    # Verification immediate sur l'ecran d'accueil, pour ne pas laisser le
    # testeur avancer (photo, choix de forme) avant de decouvrir sur
    # /api/generate que le mot de passe etait faux.
    ok = (request.get_json(silent=True) or {}).get("password") == PASSWORD
    return jsonify(ok=ok)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if request.form.get("password") != PASSWORD:
        return jsonify(error="mot de passe incorrect"), 401

    style = request.form.get("style")
    if style not in APP_STYLE_IDS:
        return jsonify(error=f"coiffure inconnue: {style!r}"), 400

    image_file = request.files.get("image")
    if not image_file:
        return jsonify(error="photo manquante"), 400

    job_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(SESSIONS_DIR, job_id)
    os.makedirs(session_dir, exist_ok=True)

    ext = os.path.splitext(image_file.filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    input_path = os.path.join(session_dir, f"input{ext}")
    image_file.save(input_path)

    job = Job(job_id, style, input_path)
    enqueue(job)
    return jsonify(job_id=job_id)


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)

    position = 0
    if job.status == "queued":
        position = queue_position(job_id)
        # temps restant avant que CE job ne demarre, plus sa propre duree estimee
        eta_seconds = position * ESTIMATED_SECONDS_PER_JOB + ESTIMATED_SECONDS_PER_JOB
    elif job.status == "running":
        # decroit reellement avec le temps ecoule -> permet un vrai compte a
        # rebours cote client plutot qu'un ETA fige tant que le statut ne
        # change pas (ce qu'on avait avant ce correctif).
        elapsed = time.time() - job.started_at
        eta_seconds = max(0, round(ESTIMATED_SECONDS_PER_JOB - elapsed))
    else:
        eta_seconds = 0

    return jsonify(
        status=job.status, queue_position=position, eta_seconds=eta_seconds, error=job.error,
    )


@app.route("/api/result/<job_id>/shape.png")
def api_result_shape(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.status != "done":
        abort(404)
    return send_from_directory(os.path.dirname(job.shape_path), os.path.basename(job.shape_path))


if __name__ == "__main__":
    print(f"-> mot de passe cabine : {PASSWORD!r} (surcharger avec CABINE_PASSWORD)")
    print("-> demarrage sur http://localhost:8080 (Ctrl+C pour arreter)")
    app.run(host="0.0.0.0", port=8080, threaded=True)

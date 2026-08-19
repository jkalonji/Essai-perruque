"""
Simule des mèches/reflets sur une photo, à partir d'une segmentation réelle
des cheveux (MediaPipe hair_segmenter), sans dépendre d'un générateur
d'images IA :
- la zone "cheveux" est détectée automatiquement sur la photo,
- un motif de mèches (bandes irrégulières, plutôt verticales) détermine
  quelles zones reçoivent la couleur et à quelle intensité,
- la couleur est appliquée en HLS en ne touchant que la teinte/saturation
  et en éclaircissant légèrement la luminosité déjà présente : les
  ombres/reflets réels de la photo restent visibles, donc le rendu suit la
  vraie texture des cheveux au lieu d'un aplat de peinture.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Par defaut tourne sur la photo brute, mais accepte n'importe quelle image
# en argument -> sert aussi a poster-traiter une image DEJA generee par
# run_kontext.py (couleur decouplee de la forme, cf. specs/cabine-coiffure-ia.html
# section 07 : la forme passe par FLUX, la couleur est un post-traitement Lab
# instantane applique ensuite, ici, sur le resultat FLUX plutot que sur la
# photo brute).
# Argument 2 optionnel : id d'un seul look (parmi LOOKS/FULL_COLORS) a
# produire, au lieu de tous -> mode utilise par cabine_server.py (couleur
# choisie par le testeur, un seul rendu a la fois, pas besoin des autres).
if len(sys.argv) > 1:
    SRC = sys.argv[1]
    # Image explicite -> ecrit a cote d'elle (pas dans results/) : c'est ce
    # chemin que prend cabine_server.py sur une image de
    # server_sessions/<job_id>/, ou une image de results/ passee a la main.
    OUT_DIR = os.path.dirname(SRC) or "."
else:
    SRC = "images/moi.jpg"
    OUT_DIR = "results"
ONLY_LOOK = sys.argv[2] if len(sys.argv) > 2 else None
MODEL = "models/hair_segmenter.tflite"
os.makedirs(OUT_DIR, exist_ok=True)
# Prefixe de sortie derive du nom du fichier d'entree (sans extension) ->
# evite d'ecraser miel_caramel.jpg/reflets_cuivres.jpg a chaque run quand on
# enchaine plusieurs formes FLUX differentes (ex. kontext_frange_xxx.png ->
# kontext_frange_xxx_miel_caramel.jpg).
OUT_PREFIX = os.path.splitext(os.path.basename(SRC))[0]

# Mèches/reflets partiels (densite de couverture basse, cf. streak_pattern) --
# usage exploratoire d'origine, gardes pour l'usage CLI manuel (README
# "Tester une couleur / des mèches").
LOOKS = [
    {"id": "miel_caramel", "label": "Mèches miel / caramel", "color_bgr": (92, 163, 204), "lighten": 55, "density": 0.3},
    {"id": "reflets_cuivres", "label": "Reflets cuivrés / roux", "color_bgr": (58, 92, 185), "lighten": 40, "density": 0.3},
]

# Couleurs pleines (densite de couverture haute -> quasi toute la chevelure,
# pas juste des mèches) -- les 3 pastilles de couleur de la cabine coiffure
# IA (specs/cabine-coiffure-ia.html section 07), appliquees apres la forme
# generee par FLUX. "lighten" monte avec la clarte visee car la photo source
# part de cheveux noirs.
FULL_COLORS = [
    {"id": "noir", "label": "Noir", "color_bgr": (12, 12, 14), "lighten": 2, "density": 0.92},
    {"id": "brun_fonce", "label": "Brun foncé", "color_bgr": (24, 32, 46), "lighten": 15, "density": 0.92},
    {"id": "brun", "label": "Brun", "color_bgr": (33, 55, 92), "lighten": 40, "density": 0.92},
    {"id": "blond", "label": "Blond", "color_bgr": (90, 150, 197), "lighten": 95, "density": 0.92},
]
# "noir" est la seule couleur exposee dans la cabine (cabine_server.py) :
# decision prise apres comparaison visuelle de "brun_fonce" vs un vrai noir
# neutre sur un rendu FLUX reel (session server_sessions/6c3538cce5c1) -> les
# deux fonctionnent (pipeline Lab, ne touche jamais au visage), "noir" rend
# une teinte neutre sans le leger cote chaud de brun_fonce, et correspond au
# rendu par defaut le plus courant pour des mèches. brun_fonce/brun/blond
# restent utilisables en CLI manuel (cf. README "Tester une couleur").

ALL_LOOKS = {look["id"]: look for look in LOOKS + FULL_COLORS}


def get_hair_mask(bgr):
    base_options = mp_python.BaseOptions(model_asset_path=MODEL)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view().squeeze()
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    # ne garde que la plus grosse zone (retire les faux positifs isolés)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask_u8 = np.where(labels == biggest, 255, 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(mask_u8, (15, 15), 0).astype(np.float32) / 255.0

    # Garde-fou : sur certaines coiffures generees par FLUX (constate sur un
    # bowl-cut/frange bien plaque), le segmenter ne reconnait qu'une infime
    # partie des cheveux comme telle (texture/eclairage inhabituels pour un
    # modele entraine sur photos reelles) -> le rendu colore serait alors
    # quasi invisible sans que rien ne l'indique. On le signale plutot que
    # de laisser un resultat silencieusement casse.
    coverage_pct = 100 * float((mask_u8 > 0).sum()) / mask_u8.size
    if coverage_pct < 2.0:
        print(f"!! ATTENTION: masque de cheveux suspect ({coverage_pct:.2f}% de l'image) "
              f"-> la segmentation a probablement echoue sur cette image, le rendu colore "
              f"risque d'etre quasi invisible.")
    return alpha


def streak_pattern(h, w, seed=0, density=0.3, edge=0.10):
    # cheveux bouclés/afro : les mèches n'y forment pas de longues bandes
    # rectilignes (comme sur cheveux lisses) mais des touffes dispersées.
    # On combine deux échelles de bruit (larges zones + texture fine) pour
    # obtenir un effet tacheté organique plutôt que 1-2 gros blocs de couleur.
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(
        rng.random((h // 55 + 2, w // 45 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    )
    fine = cv2.resize(
        rng.random((h // 16 + 2, w // 14 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    )
    noise = 0.6 * coarse + 0.4 * fine
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=6, sigmaY=9)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    # bascule doux autour du seuil de densité voulu -> mélange de mèches nettes et fondues
    weight = np.clip((noise - (1 - density) + edge) / (2 * edge), 0, 1)
    return weight


def apply_look(bgr, hair_alpha, color_bgr, lighten, seed, density=0.3, edge=0.10):
    h, w = bgr.shape[:2]
    streaks = streak_pattern(h, w, seed=seed, density=density, edge=edge)
    weight = (hair_alpha * streaks).astype(np.float32)[..., None]  # HxWx1

    # Lab : a/b sont des axes de chrominance linéaires (pas d'effet "arc-en-ciel"
    # comme avec une teinte HLS qu'on interpole partiellement), L reste la
    # luminosité déjà présente sur la photo -> on la réhausse juste un peu.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = np.uint8([[list(color_bgr)]])
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    out_lab = lab.copy()
    out_lab[..., 0] = np.clip(lab[..., 0] + weight[..., 0] * lighten, 0, 255)
    out_lab[..., 1] = lab[..., 1] * (1 - weight[..., 0]) + target_lab[1] * weight[..., 0]
    out_lab[..., 2] = lab[..., 2] * (1 - weight[..., 0]) + target_lab[2] * weight[..., 0]

    out_bgr = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out_bgr


def label_strip(images_bgr, labels, out_path):
    imgs = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images_bgr]
    w, h = imgs[0].size
    pad, bar = 8, 44
    strip = Image.new("RGB", (w * len(imgs) + pad * (len(imgs) + 1), h + bar + pad * 2), "#16151c")
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    x = pad
    for im, label in zip(imgs, labels):
        strip.paste(im, (x, bar + pad))
        draw.text((x + 6, 10), label, fill="#f2f0f5", font=font)
        x += w + pad
    strip.save(out_path)


def main():
    if ONLY_LOOK and ONLY_LOOK not in ALL_LOOKS:
        sys.exit(f"Couleur inconnue: {ONLY_LOOK!r}. Choix possibles: {', '.join(ALL_LOOKS)}")

    bgr = cv2.imread(SRC)
    if bgr is None:
        raise FileNotFoundError(SRC)
    hair_alpha = get_hair_mask(bgr)

    if ONLY_LOOK:
        # Mode "un seul look" : pas de mire de comparaison, juste le rendu
        # demande -> rapide, c'est le chemin pris par cabine_server.py quand
        # le testeur choisit une couleur sur un resultat FLUX deja genere.
        look = ALL_LOOKS[ONLY_LOOK]
        out = apply_look(
            bgr, hair_alpha, look["color_bgr"], look["lighten"], seed=1,
            density=look.get("density", 0.3),
        )
        path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{look['id']}.jpg")
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("->", path)
        return

    cv2.imwrite(os.path.join(OUT_DIR, "_debug_hair_mask.png"), (hair_alpha * 255).astype(np.uint8))

    outputs = [bgr]
    labels = ["Original"]
    for i, look in enumerate(LOOKS):
        out = apply_look(
            bgr, hair_alpha, look["color_bgr"], look["lighten"], seed=i + 1,
            density=look.get("density", 0.3),
        )
        path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{look['id']}.jpg")
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("->", path)
        outputs.append(out)
        labels.append(look["label"])

    comparaison_path = os.path.join(OUT_DIR, f"{OUT_PREFIX}_comparaison.jpg")
    label_strip(outputs, labels, comparaison_path)
    print("-> comparaison:", comparaison_path)


if __name__ == "__main__":
    main()

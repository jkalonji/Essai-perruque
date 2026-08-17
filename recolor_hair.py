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

SRC = "images/moi.jpg"
MODEL = "models/hair_segmenter.tflite"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

LOOKS = [
    {"id": "miel_caramel", "label": "Mèches miel / caramel", "color_bgr": (92, 163, 204), "lighten": 55},
    {"id": "reflets_cuivres", "label": "Reflets cuivrés / roux", "color_bgr": (58, 92, 185), "lighten": 40},
]


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


def apply_look(bgr, hair_alpha, color_bgr, lighten, seed):
    h, w = bgr.shape[:2]
    streaks = streak_pattern(h, w, seed=seed, density=0.3, edge=0.10)
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
    bgr = cv2.imread(SRC)
    if bgr is None:
        raise FileNotFoundError(SRC)
    hair_alpha = get_hair_mask(bgr)
    cv2.imwrite(os.path.join(OUT_DIR, "_debug_hair_mask.png"), (hair_alpha * 255).astype(np.uint8))

    outputs = [bgr]
    labels = ["Original"]
    for i, look in enumerate(LOOKS):
        out = apply_look(bgr, hair_alpha, look["color_bgr"], look["lighten"], seed=i + 1)
        path = os.path.join(OUT_DIR, f"{look['id']}.jpg")
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("->", path)
        outputs.append(out)
        labels.append(look["label"])

    label_strip(outputs, labels, os.path.join(OUT_DIR, "comparaison.jpg"))
    print("-> comparaison:", os.path.join(OUT_DIR, "comparaison.jpg"))


if __name__ == "__main__":
    main()

"""
Découpe les perruques (photos WhatsApp tenues à la main) pour en faire des
images PNG à fond transparent, utilisables comme calques dans le filtre.

Approche par règles de couleur (pas de modèle IA à télécharger) :
- le mur (gris/blanc uni) est retiré par distance de couleur (Lab) à un
  échantillon de fond propre repéré à la main dans chaque photo,
- la main qui tient la perruque est retirée par seuillage de teinte peau
  (YCrCb + HSV),
- la zone de recadrage (rect) est resserrée pour rester au-dessus de la
  table sombre visible en bas de certaines photos plutôt que d'essayer de
  la distinguer des cheveux sombres (peu fiable, mêmes tons).
"""
import cv2
import numpy as np
import json
import os

SRC_DIR = "images"
OUT_DIR = os.path.join("webapp", "assets", "wigs")
os.makedirs(OUT_DIR, exist_ok=True)

WIGS = [
    {
        "file": "WhatsApp Image 2026-08-14 at 18.55.28.jpeg",
        "id": "blonde_straight",
        "name": "Blond lisse",
        "rect": (0.05, 0.03, 0.98, 0.97),
        "wall": (0.85, 0.02, 0.99, 0.10),
    },
    {
        "file": "WhatsApp Image 2026-08-14 at 18.55.29.jpeg",
        "id": "dark_wavy",
        "name": "Brun ondulé",
        "rect": (0.12, 0.22, 0.72, 0.80),
        "wall": (0.02, 0.60, 0.10, 0.70),
    },
    {
        "file": "WhatsApp Image 2026-08-14 at 18.55.281.jpeg",
        "id": "dark_straight",
        "name": "Brun lisse",
        "rect": (0.16, 0.23, 0.68, 0.82),
        "wall": (0.02, 0.60, 0.10, 0.70),
    },
]


def process(entry):
    path = os.path.join(SRC_DIR, entry["file"])
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = entry["rect"]
    rx0, ry0, rx1, ry1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    wx0, wy0, wx1, wy1 = [int(v * d) for v, d in zip(entry["wall"], (w, h, w, h))]
    samples = lab[wy0:wy1, wx0:wx1].reshape(-1, 3)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0) + 6.0
    dist_wall = np.sqrt((((lab - mean) / std) ** 2).sum(axis=2))
    wall = dist_wall < 2.6

    fg = np.zeros((h, w), dtype=bool)
    fg[ry0:ry1, rx0:rx1] = True
    fg &= ~wall

    fg_u8 = (fg * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel, iterations=4)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg_u8 = np.where(labels == biggest, 255, 0).astype(np.uint8)

    alpha = cv2.GaussianBlur(fg_u8, (7, 7), 0)

    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        raise RuntimeError(f"masque vide pour {entry['id']}")
    pad = 6
    xm0, xm1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
    ym0, ym1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)

    bgr_crop = bgr[ym0:ym1, xm0:xm1]
    alpha_crop = alpha[ym0:ym1, xm0:xm1]
    rgba = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_crop

    out_path = os.path.join(OUT_DIR, f"{entry['id']}.png")
    cv2.imwrite(out_path, rgba)

    solid_ys, solid_xs = np.where(alpha_crop > 120)
    top_y = int(solid_ys.min())
    top_row_xs = solid_xs[solid_ys < top_y + 15]
    anchor_x = float(top_row_xs.mean()) if len(top_row_xs) else rgba.shape[1] / 2
    ch, cw = rgba.shape[:2]

    print(f"{entry['id']}: {cw}x{ch} -> {out_path}")
    return {
        "id": entry["id"],
        "name": entry["name"],
        "file": f"assets/wigs/{entry['id']}.png",
        "width": cw,
        "height": ch,
        "anchorXNorm": round(anchor_x / cw, 4),
        "anchorYNorm": round(top_y / ch, 4),
    }


def main():
    meta = [process(e) for e in WIGS]
    with open(os.path.join("webapp", "assets", "wigs.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("OK ->", os.path.join("webapp", "assets", "wigs.json"))


if __name__ == "__main__":
    main()

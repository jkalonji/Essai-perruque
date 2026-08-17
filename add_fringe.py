"""
Ajoute une frange complète (bouclée, comme le reste de la chevelure) sur la
photo, puis applique une des couleurs déjà choisies (miel/caramel ou
cuivré/roux) dessus.

Pas de générateur d'images IA disponible : la frange est reconstituée à
partir des VRAIS cheveux de la photo (extension verticale de la texture
juste au-dessus de la racine des cheveux actuelle, colonne par colonne),
puis recollée sur le front avec un fondu (bords adoucis). Le résultat est
donc plus artisanal qu'un rendu génératif, mais garde la vraie texture
bouclée et le vrai éclairage de la photo.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from recolor_hair import get_hair_mask, LOOKS, label_strip

SRC = "images/moi.jpg"
FACE_MODEL = "models/face_landmarker.task"
OUT_DIR = "results"

BROW_IDX = [105, 334, 336, 107]
LEFT_TEMPLE, RIGHT_TEMPLE = 127, 356


def get_face_points(bgr):
    base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise RuntimeError("Aucun visage détecté sur la photo.")
    h, w = bgr.shape[:2]
    lm = result.face_landmarks[0]
    P = lambda i: (lm[i].x * w, lm[i].y * h)
    brow_y = float(np.mean([P(i)[1] for i in BROW_IDX]))
    left_x = P(LEFT_TEMPLE)[0]
    right_x = P(RIGHT_TEMPLE)[0]
    face_top_y = P(10)[1]
    chin_y = P(152)[1]
    return {
        "brow_y": brow_y,
        "left_x": left_x,
        "right_x": right_x,
        "face_height": chin_y - face_top_y,
    }


def wavy_line(n, base_y, amplitude, seed):
    rng = np.random.default_rng(seed)
    small = rng.random(max(n // 25, 6)).astype(np.float32)
    line = cv2.resize(small.reshape(1, -1), (n, 1), interpolation=cv2.INTER_LINEAR).flatten()
    line = cv2.GaussianBlur(line.reshape(1, -1), (0, 0), sigmaX=n * 0.03).flatten()
    line = (line - line.min()) / (line.max() - line.min() + 1e-6) - 0.5
    return base_y + line * 2 * amplitude


def build_fringe(bgr, hair_mask_bool, pts, seed=0):
    h, w = bgr.shape[:2]
    left_x = int(max(pts["left_x"], 0))
    right_x = int(min(pts["right_x"], w - 1))
    margin = 0.08 * pts["face_height"]
    amplitude = 0.06 * pts["face_height"]
    fringe_bottom = wavy_line(right_x - left_x, pts["brow_y"] - margin, amplitude, seed)
    top_limit = int(pts["brow_y"] - 0.9 * pts["face_height"])

    # racine actuelle des cheveux à chaque colonne (bord bas du masque de cheveux)
    y0_per_col = np.full(right_x - left_x, -1, dtype=np.int32)
    for i, x in enumerate(range(left_x, right_x)):
        col = hair_mask_bool[top_limit : int(fringe_bottom[i]), x]
        hair_ys = np.nonzero(col)[0]
        if len(hair_ys) > 0:
            y0_per_col[i] = top_limit + int(hair_ys.max())

    valid = y0_per_col >= 0
    if not valid.any():
        raise RuntimeError("Racine des cheveux introuvable sur la largeur de la frange.")
    # comble les rares colonnes sans détection en recopiant la colonne valide la plus proche
    idx = np.arange(len(y0_per_col))
    y0_per_col[~valid] = np.interp(idx[~valid], idx[valid], y0_per_col[valid]).astype(np.int32)

    # une seule zone source de texture (au-dessus de la racine moyenne), qu'on
    # étire verticalement colonne par colonne -> pas de répétition périodique,
    # la cohérence horizontale entre colonnes voisines reste naturelle.
    patch_h = 130
    patch_top = int(np.clip(np.mean(y0_per_col) - patch_h, top_limit, h - patch_h - 1))
    source_patch = bgr[patch_top : patch_top + patch_h, left_x:right_x].astype(np.float32)

    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    new_area = np.zeros((h, w), dtype=bool)
    for i, x in enumerate(range(left_x, right_x)):
        y0 = int(y0_per_col[i])
        col_bottom = int(fringe_bottom[i])
        if col_bottom <= y0:
            continue
        ys = np.arange(y0, col_bottom)
        # coordonnées RELATIVES à source_patch (dont la ligne 0 = patch_top dans l'image d'origine)
        src_y = (ys - y0) / max(col_bottom - y0, 1) * (patch_h - 1)
        map_y[ys, x] = src_y
        map_x[ys, x] = i  # colonne dans source_patch (largeur = right_x-left_x)
        new_area[ys, x] = True

    # remap uniquement sur la largeur de la frange (source_patch), le reste garde l'image d'origine
    filled = bgr.copy()
    remapped = cv2.remap(
        source_patch, map_x[:, left_x:right_x], map_y[:, left_x:right_x], interpolation=cv2.INTER_LINEAR
    )
    filled[:, left_x:right_x] = np.where(
        new_area[:, left_x:right_x, None], remapped, bgr[:, left_x:right_x]
    ).astype(np.uint8)

    return filled, new_area, top_limit


def composite_with_feather(original, filled, new_area, feather=9):
    alpha = cv2.GaussianBlur(new_area.astype(np.float32), (0, 0), sigmaX=feather)
    alpha = np.clip(alpha / max(alpha.max(), 1e-6), 0, 1) if alpha.max() > 0 else alpha
    # on garde alpha=1 net au coeur de la zone remplie, adouci seulement aux bords
    core = cv2.erode(new_area.astype(np.uint8), np.ones((5, 5), np.uint8))
    alpha = np.maximum(alpha, core.astype(np.float32))
    alpha3 = alpha[..., None]
    return (original.astype(np.float32) * (1 - alpha3) + filled.astype(np.float32) * alpha3).astype(np.uint8)


def color_region(bgr, weight, color_bgr, lighten):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = np.uint8([[list(color_bgr)]])
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    w3 = weight[..., None]
    out = lab.copy()
    out[..., 0] = np.clip(lab[..., 0] + weight * lighten, 0, 255)
    out[..., 1] = lab[..., 1] * (1 - w3[..., 0]) + target_lab[1] * w3[..., 0]
    out[..., 2] = lab[..., 2] * (1 - w3[..., 0]) + target_lab[2] * w3[..., 0]
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main():
    bgr = cv2.imread(SRC)
    if bgr is None:
        raise FileNotFoundError(SRC)
    h, w = bgr.shape[:2]

    hair_alpha = get_hair_mask(bgr)
    hair_mask_bool = hair_alpha > 0.5

    pts = get_face_points(bgr)
    filled, new_area, top_limit = build_fringe(bgr, hair_mask_bool, pts, seed=3)
    composited = composite_with_feather(bgr, filled, new_area, feather=6)

    cv2.imwrite(f"{OUT_DIR}/_debug_frange_brute.jpg", composited)

    # zone de recoloration : la frange + un léger débord dans les cheveux
    # existants juste au-dessus, pour que la couleur ne s'arrête pas net
    recolor_zone = cv2.dilate(new_area.astype(np.uint8), np.ones((25, 25), np.uint8)) & (
        hair_mask_bool | new_area
    )
    weight = cv2.GaussianBlur(recolor_zone.astype(np.float32), (0, 0), sigmaX=10)
    weight = np.clip(weight / max(weight.max(), 1e-6), 0, 1) * 0.9

    outputs = [bgr]
    labels = ["Original"]
    for look in LOOKS:
        colored = color_region(composited, weight, look["color_bgr"], look["lighten"])
        path = f"{OUT_DIR}/frange_{look['id']}.jpg"
        cv2.imwrite(path, colored, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print("->", path)
        outputs.append(colored)
        labels.append(f"Frange — {look['label']}")

    label_strip(outputs, labels, f"{OUT_DIR}/comparaison_frange.jpg")
    print("-> comparaison:", f"{OUT_DIR}/comparaison_frange.jpg")


if __name__ == "__main__":
    main()

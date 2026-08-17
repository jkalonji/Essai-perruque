"""
Mesure la luminance globale et la luminance de la zone "cheveux" (via le
même segmenteur MediaPipe que recolor_hair.py) d'une image de sortie
HairFastGAN, pour comparer objectivement les essais de référence de forme.
"""
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = "models/hair_segmenter.tflite"


def get_hair_mask_bool(bgr):
    base_options = mp_python.BaseOptions(model_asset_path=MODEL)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view().squeeze()
    return mask > 0


def luminance(bgr):
    b, g, r = bgr[..., 0].astype(np.float32), bgr[..., 1].astype(np.float32), bgr[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def main(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    lum = luminance(bgr)
    print(f"{path}")
    print(f"  luminance globale : {lum.mean():.1f} / 255")
    try:
        mask = get_hair_mask_bool(bgr)
        if mask.sum() > 0:
            print(f"  luminance zone cheveux : {lum[mask].mean():.1f} / 255  ({mask.sum()} px)")
        else:
            print("  aucun pixel cheveux détecté")
    except Exception as e:
        print("  (segmentation cheveux impossible:", e, ")")


if __name__ == "__main__":
    main(sys.argv[1])

"""
Assemble une planche comparative des rendus de forme HairFastGAN
(results/forme_*.png), éclaircis pour rendre la silhouette lisible malgré
la photo source à contre-jour (cheveux/visage naturellement sombres).
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps

OUT_DIR = "results"

SHAPES = [
    ("forme_lisse", "Lisse"),
    ("forme_ondulee", "Ondulée"),
    ("forme_tres_ondulee", "Très ondulée"),
    ("forme_curly", "Curly"),
]


def brighten(im):
    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageEnhance.Brightness(im).enhance(1.8)
    return ImageEnhance.Contrast(im).enhance(1.15)


def label_strip(images, labels, out_path):
    w, h = images[0].size
    pad, bar = 8, 44
    strip = Image.new("RGB", (w * len(images) + pad * (len(images) + 1), h + bar + pad * 2), "#16151c")
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    x = pad
    for im, label in zip(images, labels):
        strip.paste(im, (x, bar + pad))
        draw.text((x + 6, 10), label, fill="#f2f0f5", font=font)
        x += w + pad
    strip.save(out_path)


def main():
    images, labels = [], []
    for name, label in SHAPES:
        path = os.path.join(OUT_DIR, f"{name}.png")
        im = brighten(Image.open(path).convert("RGB"))
        images.append(im)
        labels.append(label)
    out_path = os.path.join(OUT_DIR, "comparaison_formes.jpg")
    label_strip(images, labels, out_path)
    print("->", out_path)


if __name__ == "__main__":
    main()

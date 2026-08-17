"""
Corrige l'exposition des rendus HairFastGAN (results/forme_*.png), qui
ressortent systématiquement trop sombres sur cheveux/peau foncés (biais
connu du StyleGAN2 entraîné sur FFHQ, indépendant de la qualité de la
photo source — cf. mesures faites en session).

CLAHE (contraste local adaptatif) sur le canal L (Lab) plutôt qu'un
simple gamma/brightness : un gamma uniforme délave toute l'image parce
que l'info est comprimée sur une plage de valeurs très étroite (~4-25
sur 255) ; CLAHE relève le contraste PAR ZONE, ce qui fait ressortir les
variations déjà présentes dans le rendu (et donc les reflets qui suivent
la forme des mèches) sans toucher la teinte (canaux a/b inchangés).
"""
import os
import cv2

RESULTS_DIR = "results"
CLIP_LIMIT = 3.0
TILE_SIZE = 8


def relight(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(TILE_SIZE, TILE_SIZE))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def main(names=None):
    names = names or ["forme_lisse", "forme_ondulee", "forme_tres_ondulee", "forme_curly"]
    for name in names:
        src = os.path.join(RESULTS_DIR, f"{name}.png")
        if not os.path.exists(src):
            print("!! introuvable:", src)
            continue
        out = relight(cv2.imread(src))
        dst = os.path.join(RESULTS_DIR, f"{name}_clair.png")
        cv2.imwrite(dst, out)
        print("->", dst)


if __name__ == "__main__":
    main()

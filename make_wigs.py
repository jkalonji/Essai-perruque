"""
Génère des assets de perruque prêts pour l'overlay temps réel,
à partir des textures capillaires réelles des photos produit.

Sortie : PNG/WebP RGBA 1000x1500 avec
  - silhouette portable vue de face (calotte + panneaux latéraux + longueur)
  - ouverture visage
  - bords vaporeux (mèches) via bruit vertical
  - ombre de contact sous la ligne frontale
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, gaussian_filter

U = '/mnt/user-data/uploads/'
W, H = 1000, 1500
CX = W / 2

STYLES = {
    'blonde': dict(
        src=U + '1786732173284_WhatsApp_Image_2026-08-14_at_18_55_28.jpeg',
        crop=(600, 980, 985, 1660),
        wave_amp=6, wave_period=520, frizz=0.35,
        root_dark=0.78, root_warm=(1.06, 0.95, 0.80), root_depth=210,
        length=1440, taper=0.90,
    ),
    'wave': dict(
        src=U + '1786732173284_WhatsApp_Image_2026-08-14_at_18_55_30.jpeg',
        crop=(540, 800, 980, 1460),
        wave_amp=17, wave_period=330, frizz=0.55,
        root_dark=0.80, root_warm=(1.0, 0.98, 0.96), root_depth=200,
        length=1450, taper=0.93,
    ),
    'straight': dict(
        src=U + '1786732173284_WhatsApp_Image_2026-08-14_at_18_55_31.jpeg',
        crop=(400, 640, 790, 1280),
        wave_amp=4, wave_period=600, frizz=0.30,
        root_dark=0.82, root_warm=(1.0, 0.99, 0.97), root_depth=190,
        length=1400, taper=0.88,
    ),
    'natural': dict(
        src=U + '1786732173285_WhatsApp_Image_2026-08-14_at_18_55_32.jpeg',
        crop=(840, 1000, 1330, 1680),
        wave_amp=22, wave_period=250, frizz=0.85,
        root_dark=0.86, root_warm=(1.0, 1.0, 1.0), root_depth=230,
        length=1380, taper=1.02,
    ),
}


# ---------------------------------------------------------------- silhouette
def half_width_profile(kind, length, taper):
    """Demi-largeur extérieure de la chevelure, par ligne."""
    y = np.arange(H, dtype=float)
    # calotte : arc de cercle (crâne)
    crown_cy, crown_r = 262.0, 216.0
    ow = np.zeros(H)
    top = crown_cy - crown_r  # y=50
    cap = y < crown_cy
    inside = cap & (y >= top)
    ow[inside] = np.sqrt(np.maximum(crown_r ** 2 - (crown_cy - y[inside]) ** 2, 0))
    # corps : élargissement puis longueur
    pts_y = [262, 380, 520, 700, 900, 1100, length - 120, length, length + 60]
    pts_w = [216, 254, 288, 318, 342, 354, 344 * taper, 306 * taper, 0]
    body = y >= 262
    ow[body] = np.interp(y[body], pts_y, pts_w)
    ow[y > length + 60] = 0
    return ow


def hole_profile():
    """Demi-largeur de l'ouverture visage/buste, par ligne (0 = pas d'ouverture)."""
    y = np.arange(H, dtype=float)
    iw = np.zeros(H)
    # ligne frontale : demi-ellipse
    hair_cy, hair_rx, hair_ry = 372.0, 162.0, 178.0
    top = hair_cy - hair_ry  # y=145
    arc = (y >= top) & (y < hair_cy)
    iw[arc] = hair_rx * np.sqrt(np.maximum(1 - ((hair_cy - y[arc]) / hair_ry) ** 2, 0))
    # visage -> mâchoire -> menton -> cou -> buste
    pts_y = [372, 470, 560, 640, 700, 745, 800, 870, 960, 1100, 1300, 1500]
    pts_w = [162, 166, 158, 140, 114, 92, 84, 108, 152, 190, 214, 228]
    low = y >= 372
    iw[low] = np.interp(y[low], pts_y, pts_w)
    return iw


def build_masks(cfg):
    y = np.arange(H, dtype=float)[:, None]
    x = np.arange(W, dtype=float)[None, :]

    ow = half_width_profile('', cfg['length'], cfg['taper'])[:, None]
    iw = hole_profile()[:, None]

    # ondulation : phases différentes à gauche/droite (asymétrie naturelle)
    p = cfg['wave_period']
    a = cfg['wave_amp']
    growth = np.clip((y - 250) / 600, 0, 1)  # pas d'ondulation sur la calotte
    wl = a * np.sin(y / p * 2 * np.pi + 0.7) * growth
    wr = a * np.sin(y / p * 2 * np.pi + 2.9) * growth

    left = CX - (ow + wl)
    right = CX + (ow + wr)
    outer = (x >= left) & (x <= right)

    hole_top = 194
    inner = (np.abs(x - CX) <= iw) & (y >= hole_top)

    hair = (outer & ~inner).astype(np.float32)
    hole = inner.astype(np.float32)
    return hair, hole


def wispy_alpha(hair, cfg, seed=0):
    """Bords vaporeux : la distance au bord est rongée par un bruit en mèches."""
    rng = np.random.default_rng(seed)
    d_in = distance_transform_edt(hair)

    # bruit en stries verticales (mèches)
    n = rng.random((H, W)).astype(np.float32)
    n = gaussian_filter1d(n, sigma=26, axis=0)   # très lissé verticalement
    n = gaussian_filter1d(n, sigma=1.6, axis=1)  # peu lissé horizontalement
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)

    y = np.arange(H, dtype=np.float32)[:, None]
    # plus vaporeux vers les pointes
    feather = 13 + 44 * np.clip((y - 500) / 900, 0, 1)
    bite = (16 + 74 * np.clip((y - 400) / 1000, 0, 1)) * cfg['frizz']
    bite = bite + 11 * np.exp(-((y - 245.0) / 95.0) ** 2)

    a = np.clip((d_in - n * bite) / feather, 0, 1)
    a = np.clip(a * 1.12, 0, 1)
    a = gaussian_filter(a, sigma=0.7)
    return a.astype(np.float32)


# ------------------------------------------------------------------ texture
def hair_field(cfg):
    im = Image.open(cfg['src']).convert('RGB').crop(cfg['crop'])
    tw, th = im.size
    # tuile étirée sur toute la hauteur
    TH = H + 460
    tile = im.resize((max(int(tw * TH / th), 260), TH), Image.LANCZOS)
    t = np.asarray(tile, dtype=np.float32)
    tw = t.shape[1]

    # pavage en miroir sur la largeur, avec décalages verticaux pour casser la répétition
    cols, off = [], 0
    rng = np.random.default_rng(7)
    flip = False
    while off < W + tw:
        piece = t[:, ::-1] if flip else t
        shift = int(rng.integers(0, 460))
        cols.append(piece[shift:shift + H])
        flip = not flip
        off += tw
    field = np.concatenate(cols, axis=1)[:, :W]

    # racines : assombrissement + tonalité chaude sur la calotte
    y = np.arange(H, dtype=np.float32)[:, None, None]
    rd = cfg['root_depth']
    k = np.clip(1 - y / rd, 0, 1) ** 1.4
    warm = np.array(cfg['root_warm'], dtype=np.float32)[None, None, :]
    field = field * (1 - k) + field * k * cfg['root_dark'] * warm

    # volume : léger dégradé latéral (les côtés reçoivent moins de lumière)
    x = np.arange(W, dtype=np.float32)[None, :, None]
    side = 1 - 0.16 * np.abs(x - CX) / CX
    field = field * side

    return np.clip(field, 0, 255)


def parting(field, cfg):
    """Raie centrale discrète sur la calotte."""
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    band = np.exp(-((x - CX) ** 2) / (2 * 9.0 ** 2)) * np.clip((200 - y) / 160, 0, 1)
    return field * (1 - 0.45 * band[..., None])


# ------------------------------------------------------------ ombre portée
def contact_shadow(hole):
    d = distance_transform_edt(hole)  # distance au bord, à l'intérieur de l'ouverture
    y = np.arange(H, dtype=np.float32)[:, None]
    fade = np.clip(1 - (y - 200) / 620, 0, 1) ** 1.3  # s'estompe sous les yeux
    s = np.exp(-d / 38.0) * 0.42 * fade * hole
    return gaussian_filter(s, sigma=6).astype(np.float32)


# ------------------------------------------------------------------- rendu
def render(name, cfg):
    hair, hole = build_masks(cfg)
    alpha = wispy_alpha(hair, cfg, seed=abs(hash(name)) % 999)
    field = parting(hair_field(cfg), cfg)
    shadow = contact_shadow(hole)

    total = np.clip(alpha + shadow * (1 - alpha), 0, 1)
    rgb = field * alpha[..., None] / np.maximum(total, 1e-5)[..., None]
    rgb = np.clip(rgb, 0, 255)

    out = np.dstack([rgb, total * 255]).astype(np.uint8)
    img = Image.fromarray(out, 'RGBA')
    img.save(f'assets/wig_{name}.webp', quality=86, method=6)
    img.resize((500, 750), Image.LANCZOS).save(f'assets/prev_{name}.png')
    print(name, 'ok')


def thumbs():
    """Vignettes : vraies photos produit, recadrées en mèche verticale."""
    src = {
        'blonde': (U + '1786732173284_WhatsApp_Image_2026-08-14_at_18_55_28.jpeg', (430, 200, 1130, 1700)),
        'wave': (U + '1786732173284_WhatsApp_Image_2026-08-14_at_18_55_30.jpeg', (330, 120, 1110, 1790)),
        'straight': (U + '1786732173284_WhatsApp_Image_2026-08-14_at_18_55_31.jpeg', (180, 130, 940, 1760)),
        'natural': (U + '1786732173285_WhatsApp_Image_2026-08-14_at_18_55_32.jpeg', (330, 330, 1330, 2040)),
    }
    for k, (f, box) in src.items():
        im = Image.open(f).convert('RGB').crop(box).resize((168, 360), Image.LANCZOS)
        im.save(f'assets/thumb_{k}.webp', quality=80, method=6)


import os
os.makedirs('assets', exist_ok=True)
for k, v in STYLES.items():
    render(k, v)
thumbs()
print('\n--- tailles ---')
for f in sorted(os.listdir('assets')):
    print(f, os.path.getsize('assets/' + f) // 1024, 'ko')

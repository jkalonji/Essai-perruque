# Essai de perruque — filtre webcam

Un mini "filtre Snapchat" pour essayer virtuellement les 3 perruques
photographiées dans `images/`, directement dans le navigateur via la webcam.
Tout tourne en local — aucune image de ta webcam n'est envoyée où que ce soit.

## Lancer le filtre

1. Double-clique sur **`start.bat`**.
   Ça lance un petit serveur local et ouvre `http://localhost:8000` dans ton
   navigateur (un serveur local est nécessaire : la caméra ne fonctionne pas
   en ouvrant `index.html` directement en `file://`).
2. Autorise l'accès à la caméra quand le navigateur le demande.
3. Choisis une perruque dans le panneau de droite, ajuste taille / hauteur /
   rotation avec les sliders jusqu'à ce que ça tombe bien, et prends une
   photo avec 📸.
4. `Ctrl+C` dans la fenêtre noire pour arrêter le serveur quand tu as fini.

Le premier lancement télécharge le modèle de suivi de visage de Google
(MediaPipe) — une connexion internet est donc nécessaire la première fois.
Ensuite le navigateur le garde en cache.

## Comment ça marche

- **`process_wigs.py`** : découpe les photos WhatsApp (perruque tenue à la
  main devant un mur) pour n'en garder que les cheveux, fond transparent.
  Pas de service externe : juste de la détection de couleur (OpenCV) —
  le mur est retiré par comparaison de couleur, et les rectangles de
  recadrage évitent la main et la table sombre visibles sur certaines
  photos. Résultat dans `webapp/assets/wigs/*.png` + `webapp/assets/wigs.json`.
- **`webapp/app.js`** : ouvre la webcam, utilise **MediaPipe FaceLandmarker**
  (suivi de visage en temps réel, dans le navigateur) pour repérer le haut
  du front, la largeur du visage et l'inclinaison de la tête à chaque
  image, puis positionne/redimensionne/tourne le PNG de la perruque en
  conséquence sur un `<canvas>` superposé à la vidéo.
- Les sliders (taille, hauteur, décalage, rotation) corrigent l'ajustement
  automatique perruque par perruque — le cadrage des photos sources n'étant
  pas parfaitement symétrique, un ajustement à l'œil reste nécessaire.

## Ajouter une nouvelle perruque

1. Ajoute une photo dans `images/` (perruque tenue devant un fond uni,
   si possible sans table sombre en arrière-plan).
2. Ajoute une entrée dans la liste `WIGS` de `process_wigs.py` avec :
   - `rect` : la zone (en %) qui contient les cheveux, sans la main,
   - `wall` : un petit échantillon (en %) d'une zone de mur bien unie,
     pour calibrer le retrait du fond.
3. Relance : `python process_wigs.py`
4. (Optionnel) ajoute des réglages par défaut pour cette perruque dans
   `DEFAULTS` en haut de `webapp/app.js`.

## Tester une couleur / des mèches sur ta vraie photo

Plutôt qu'un filtre webcam en direct, cette partie fonctionne en deux temps
(meilleure qualité, pas de contrainte temps réel) :

1. **Prendre la photo** : lance `start.bat`, ouvre
   `http://localhost:8000/capture.html`, prends la photo (visage dégagé,
   bien éclairé, pas de contre-jour), enregistre-la dans `images/moi.jpg`.
2. **Générer les rendus** : `python recolor_hair.py`
   → produit `results/miel_caramel.jpg`, `results/reflets_cuivres.jpg` et
   `results/comparaison.jpg` (les 3 côte à côte).

Comment ça marche : `recolor_hair.py` utilise le modèle **MediaPipe Hair
Segmenter** (téléchargé une fois dans `models/`) pour détecter précisément
tes cheveux sur la photo, génère un motif de mèches organique (pas une
teinte plate), puis recolore en espace **Lab** en ne touchant que la
teinte/saturation — la luminosité (donc les ombres et reflets réels de la
photo) est conservée et même légèrement rehaussée sur les mèches. Résultat :
la texture de tes cheveux reste visible sous la couleur, comme sur une vraie
photo retouchée plutôt qu'un calque plaqué dessus.

Pour changer les couleurs testées ou leur répartition, modifie la liste
`LOOKS` (couleur en BGR, intensité d'éclaircissement) et les paramètres
`density`/`edge` de `streak_pattern` (plus `density` est haut, plus il y a
de mèches colorées) en haut de `recolor_hair.py`, puis relance le script.

## Tester une frange entière

`python add_fringe.py` (à lancer après `recolor_hair.py`, il réutilise
`images/moi.jpg`) → produit `results/frange_miel_caramel.jpg`,
`results/frange_reflets_cuivres.jpg` et `results/comparaison_frange.jpg`.

Comment ça marche : **MediaPipe Face Landmarker** repère le haut des
sourcils et les tempes pour définir la zone de la frange (du haut du front
jusqu'à juste au-dessus des sourcils, avec un bord bas légèrement irrégulier
pour un effet "coupe" naturel plutôt qu'une ligne droite). Comme je n'ai pas
de générateur d'images IA, cette zone est remplie avec de la **vraie
texture de cheveux de la photo** (le patch juste au-dessus de la racine
actuelle, étiré verticalement colonne par colonne pour épouser la ligne de
frange), recollée avec un fondu sur les bords — puis colorée avec la même
méthode Lab que pour les mèches. C'est donc un compositing "artisanal"
(pas un rendu génératif) : la forme suit tes vrais sourcils/tempes et garde
ta vraie texture bouclée, mais un léger raccord peut rester visible à la
jonction avec le reste des cheveux.

## Rendu génératif réel (HairFastGAN, qualité "hairstyleai.ai")

`HairFastGAN/` fait tourner en local, sur ton GPU (NVIDIA RTX 5060 Ti), le
vrai modèle **HairFastGAN** (AIRI-Institute, NeurIPS 2024) — celui qui
équipe ce genre de site. Contrairement à `add_fringe.py`, ce n'est pas du
compositing : le visage est ré-encodé dans l'espace latent d'un StyleGAN2
puis regénéré en entier avec la forme et la couleur de cheveux demandées,
d'où un rendu bien plus net et cohérent (mais recadré/redimensionné en
1024×1024, format natif du modèle — donc plus proche d'une photo d'identité
que ta photo originale).

### Lancer

```bash
cd HairFastGAN
.venv/Scripts/python.exe run_local.py
```

Modifie en haut de `run_local.py` :
- `FACE` : ta photo (`../images/moi.jpg`)
- `SHAPE` : photo de référence pour la **forme** de coiffure voulue
- `LOOKS` : liste de (nom de sortie, photo de référence pour la **couleur**)

Les photos de référence utilisées sont dans `refs/` (trouvées sur Pexels,
libres de droits). Le modèle a besoin d'un visage détectable par dlib dans
chaque photo de référence — teste avec :
```python
import dlib, cv2
det = dlib.get_frontal_face_detector()
print(len(det(cv2.cvtColor(cv2.imread("refs/ta_photo.jpg"), cv2.COLOR_BGR2RGB), 1)))
```
(0 = pas de visage détecté, il en faut une autre — les gros plans/profils
extrêmes échouent souvent).

### Ce qui a été nécessaire pour que ça tourne (pour info / si tu réinstalles)

- Environnement virtuel dédié `HairFastGAN/.venv` (⚠️ ne jamais installer ces
  dépendances dans l'environnement Python global — première tentative
  ratée qui a cassé des paquets partagés, corrigée depuis).
- PyTorch CUDA 12.8 (`pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`)
  pour supporter ta carte (Blackwell, capability 12.0).
- `dlib-bin` au lieu de `dlib` (le vrai `dlib` doit se compiler avec
  CMake + un compilateur C++, absents ici).
- 4 fichiers de `models/**/stylegan2/op/{fused_act,upfirdn2d}.py` patchés
  pour utiliser leur repli pur PyTorch plutôt que de compiler une extension
  CUDA maison (qui aurait nécessité Visual Studio Build Tools + CUDA
  Toolkit complet, non installés) — légèrement plus lent, invisible ici.
- Poids du modèle (~7,2 Go) dans `HairFastGAN/pretrained_models/`,
  téléchargés depuis Hugging Face.

Poids d'origine sur https://huggingface.co/spaces/AIRI-Institute/HairFastGAN
et code sur https://github.com/AIRI-Institute/HairFastGAN — la démo
publique elle-même (et ses forks) est en panne depuis un moment (tourne
sur du CPU gratuit alors que le modèle a besoin d'un GPU), d'où le passage
en local.

## Quelle méthode utiliser ?

| Besoin | Outil |
|---|---|
| Essayer une **perruque** en direct à la webcam | `webapp/index.html` |
| Juste tester une **couleur/mèches** sur ta vraie coiffure actuelle | `recolor_hair.py` |
| Tester une **frange**, rendu rapide mais approximatif | `add_fringe.py` |
| Tester une **frange + couleur**, rendu le plus réaliste (recadré 1024×1024, GPU requis) | `HairFastGAN/run_local.py` |

## Limites (c'est un prototype simple)

- Le détourage des cheveux est fait par couleur, pas par IA — les bords sont
  propres mais pas parfaits (petit reflet de main possible en haut de
  quelques perruques).
- Le placement automatique est une approximation 2D (pas de vraie 3D) : il
  suit bien les mouvements de la tête, mais un profil très de biais peut
  décrocher un peu — les sliders compensent.

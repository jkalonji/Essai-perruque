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

## Générer une coiffure complètement différente (FLUX.1-Kontext-dev, local)

`run_kontext.py` fait tourner en local, sur ton GPU (NVIDIA RTX 5060 Ti), le
modèle **FLUX.1-Kontext-dev** (Black Forest Labs) — un éditeur d'image par
instruction texte. Contrairement à `add_fringe.py` (compositing) ou au
filtre webcam (calque plaqué), c'est une vraie régénération par un modèle
de diffusion à partir d'une instruction en langage naturel ("change
hairstyle to..."), en préservant le reste de la photo (visage, fond,
lunettes...). Le rendu le plus réaliste des méthodes du dépôt, et la seule
qui permette de décrire n'importe quelle coiffure en texte libre plutôt
qu'à partir d'une photo de référence.

### Lancer

```bash
python run_kontext.py brushing_frange
```

(sans argument → coiffure par défaut, voir `DEFAULT_STYLE` dans le fichier).
Le script lit `images/moi.jpg`, applique la coiffure demandée, et sauve le
résultat dans `results/kontext_<style>_<horodatage>.png` (jamais écrasé).

Premier lancement : le transformer (~24 Go en bf16) est téléchargé une fois
depuis Hugging Face (connexion internet nécessaire), quantifié en FP8 pour
tenir dans 16 Go de VRAM, et mis en cache localement dans
`models/flux_kontext_transformer_fp8/` — les lancements suivants sautent
cette étape et démarrent en quelques secondes. Une génération prend environ
3 minutes sur une RTX 5060 Ti 16 Go.

### Prérequis / performance

- **GPU NVIDIA avec au moins ~16 Go de VRAM** (testé sur une RTX 5060 Ti).
  Télécharger le modèle ne suffit pas : le transformer pèse ~24 Go en bf16,
  bien plus que la VRAM disponible — il doit d'abord être quantifié en FP8
  poids-seuls (torchao) pour tenir en mémoire, d'où le passage bf16 → FP8 au
  premier lancement (voir le docstring de `run_kontext.py`).
- **Pagefile Windows généreux recommandé** (32-64 Go) : si la VRAM est déjà
  quasi saturée (par ex. un autre outil GPU ouvert en parallèle, comme
  LM Studio), le driver NVIDIA bascule silencieusement une partie des
  données vers la RAM système/le pagefile au lieu d'échouer franchement — ça
  ne plante pas, mais une génération peut alors passer de ~3 minutes à
  près de 2 heures. Symptôme reconnaissable : GPU à 100% d'utilisation dans
  le gestionnaire de tâches mais consommation électrique très faible
  (~35-40 W au lieu de ~180 W) → c'est le signe que le GPU attend de la
  mémoire plutôt que de calculer. Dans ce cas, ferme les autres applications
  qui utilisent le GPU avant de lancer une génération.
- Le script mitige déjà cette contrainte mémoire par défaut : image d'entrée
  redimensionnée (`MAX_SIDE = 1024`), moins d'étapes de débruitage
  (`NUM_STEPS = 20` au lieu de 28 par défaut) et `enable_model_cpu_offload()`
  (ne garde sur le GPU que le composant du pipeline en train de travailler).

### Choisir/ajouter une coiffure

Les coiffures disponibles sont définies dans le dictionnaire `HAIRSTYLES` en
tête de `run_kontext.py` : `cle -> (prompt, LoRA optionnel)`. Pour en
ajouter une, ajoute une entrée avec une description en anglais du résultat
voulu.

Certains vocabulaires font dériver le modèle vers un visage différent (voire
un genre/âge différent) au lieu de se contenter d'éditer les cheveux —
observé et reproduit plusieurs fois pendant le réglage des prompts (ex. la
combinaison "long past the shoulders" + "face-framing" + "sleek and neatly
styled"). Une clause générique de type "Keep the same face and identity
unchanged." a été testée mais n'est pas fiable (parfois ignorée par le
modèle) : elle a été retirée. À la place, chaque prompt de `HAIRSTYLES`
mentionne explicitement le genre de la personne ("This is a man."/"This is
a woman.") quand c'est pertinent — mais ça ne garantit pas non plus la
préservation du visage : certains vocabulaires (ex. "blowout/face-framing/
sleek") font dériver le visage quel que soit le seed testé, d'autres (ex.
"curly/ringlet") sont plus proches d'un seuil où le seed peut faire basculer
le résultat dans un sens ou l'autre (voir le commentaire au-dessus de `SEED`
dans `run_kontext.py` pour le détail des tests). Un seed fixe (`SEED` dans
le fichier) permet de reproduire exactement un résultat et de distinguer un
effet du prompt d'un simple coup de chance.

### Suivre ce qui a été généré

Chaque génération est loggée dans `results/generation_log.tsv`
(`timestamp | style | fichier | lora | seed | prompt`) — pratique pour
retrouver quel prompt exact (et quel seed) a produit quelle image sans
rouvrir le log de chaque run. Fichier texte à colonnes, lisible avec
`cat`/`less`/un tableur.

## Site "Mèches" — vitrine + essai virtuel par lien (téléphone)

`cabine_server.py` sert un petit site complet à trois onglets, tous servis
par le même serveur Flask (pas de build, pas de dépendance externe) :

- **Accueil** (`/`, `site/index.html`) et **Boutique** (`/boutique`,
  `site/boutique.html`) — vitrine e-commerce de démonstration (catalogue de
  mèches factice, `site/assets/catalog.json`, 6 produits) inspirée du
  minimalisme de rains.com et gemmyo.com. Photos issues de banques **libres
  de droits, usage commercial autorisé** (Pexels License — pas d'obligation
  d'attribution, créditées quand même sur chaque fiche produit, cf. champ
  `credit` du catalogue) : ni les photos personnelles fournies par
  l'utilisateur, ni des images glanées sans licence claire sur le web n'ont
  été utilisées pour ce catalogue. **Pas de vrai panier/paiement** —
  "Ajouter au panier" affiche un message honnête plutôt que de faire
  semblant de fonctionner. À remplacer par un vrai catalogue/backend (et de
  vraies photos produit) si le projet passe en boutique fonctionnelle.
- **Essayer des mèches** (`/essayer/`, `cabine/`) — l'outil d'essai virtuel
  par IA (ex-"cabine coiffure IA") : mot de passe → photo caméra → choix
  d'une forme (5 cartes : brushing/attaché/frange/raie au milieu/bouclé) →
  attente (avec compte à rebours) → résultat. Une seule couleur (noir) est
  appliquée automatiquement en fin de génération — un essai de "couleur
  dans le prompt FLUX" (blond/brun choisis par le testeur) a été écarté :
  ça marchait pour certaines combinaisons forme+couleur mais en faisait
  dériver d'autres vers un visage complètement différent (même piège que
  documenté plus bas pour la forme seule). Le post-traitement Lab
  (`recolor_hair.py`) reste la seule méthode de couleur : il ne touche
  jamais au visage, seulement au masque cheveux détecté. Chaque fiche
  produit de la boutique pointe vers cet onglet avec la forme correspondante
  pré-sélectionnée (`/essayer/?style=...`). Spec complète de l'outil :
  `specs/cabine-coiffure-ia.html`.

Le style commun (nav, typographie, boutons) vit dans `site/assets/brand.css`,
chargé sur les trois surfaces — `cabine/style.css` ne définit plus sa propre
palette, il réutilise les variables de `brand.css`.

### Lancer en local

```bash
".venv/Scripts/python.exe" cabine_server.py
```

Ouvre <http://localhost:8080> (accueil du site). Mot de passe par défaut de
l'onglet "Essayer des mèches" : `coiffure2026` (surcharger avec la variable
d'environnement `CABINE_PASSWORD`) — l'accueil et la boutique sont publics,
seul l'essai virtuel (coûteux en GPU) est protégé. Une seule génération à la
fois : les requêtes suivantes patientent en file — pas de gestion de
plusieurs testeurs simultanés (hors scope, voir le spec §09).

La couleur est un post-traitement `recolor_hair.py` exécuté en sous-processus
avec le **Python système** (celui qui a `cv2`/`mediapipe`), pas le `.venv`
dédié à FLUX — les deux environnements sont volontairement séparés (torch/
diffusers d'un côté, opencv/mediapipe de l'autre). Si `python` ne résout pas
vers le bon interpréteur sur ta machine, surcharge avec
`CABINE_RECOLOR_PYTHON`.

### Partager par lien (tunnel)

L'accès caméra (`getUserMedia`) exige HTTPS sur mobile — un simple partage
d'IP locale ne suffit pas. [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
(installé via `winget install Cloudflare.cloudflared`) donne une URL HTTPS
publique directement :

```bash
cloudflared tunnel --url http://localhost:8080
```

Si `cloudflared` n'est pas reconnu juste après l'installation, ouvre un
nouveau terminal (le `PATH` n'est mis à jour que dans les nouvelles
fenêtres) — ou lance-le directement via son chemin complet :
`"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8080`.

⚠️ Ça expose le GPU de cette machine à quiconque a le lien (protection = le
mot de passe partagé uniquement, voir spec §06/§08) — à ne lancer qu'au
moment d'une session avec un testeur, pas à laisser tourner en continu. Le
lien meurt dès que le tunnel ou le PC s'arrête.

## Quelle méthode utiliser ?

| Besoin | Outil |
|---|---|
| Essayer une **perruque** en direct à la webcam | `webapp/index.html` |
| Juste tester une **couleur/mèches** sur ta vraie coiffure actuelle | `recolor_hair.py` |
| Tester une **frange**, rendu rapide mais approximatif | `add_fringe.py` |
| Tester une **coiffure décrite en texte libre**, rendu le plus réaliste (GPU requis) | `run_kontext.py` |
| Voir le site vitrine/boutique (démo) et l'essai virtuel **par lien**, depuis un téléphone | `cabine_server.py` + `site/` + `cabine/` |

## Limites (c'est un prototype simple)

- Le détourage des cheveux est fait par couleur, pas par IA — les bords sont
  propres mais pas parfaits (petit reflet de main possible en haut de
  quelques perruques).
- Le placement automatique est une approximation 2D (pas de vraie 3D) : il
  suit bien les mouvements de la tête, mais un profil très de biais peut
  décrocher un peu — les sliders compensent.
- `run_kontext.py` a besoin d'une photo où les cheveux sont réellement
  visibles (pas de bonnet/casquette qui les recouvre) : sans zone de
  cheveux à éditer localement, le modèle peut dériver bien au-delà de la
  coiffure (visage, tenue...).
- `run_kontext.py` ne garantit pas de préserver le visage/l'identité de la
  photo source : selon le vocabulaire du prompt (et parfois le seed), le
  modèle peut générer un visage différent — voir la section
  "Choisir/ajouter une coiffure" ci-dessus.

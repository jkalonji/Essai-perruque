# Essayage virtuel de perruques — prototype

Fichier unique, autonome : `essayage-perruques.html` (539 ko, images comprises).
Aucun serveur, aucune dépendance à installer, aucune donnée envoyée. Le flux vidéo
ne quitte jamais l'appareil.

## Mettre en ligne (pour envoyer un lien au client)

La caméra exige **HTTPS**. Ouvrir le fichier en `file://` ne marchera pas.

| Option | Marche à suivre | Durée |
|---|---|---|
| Netlify Drop | `app.netlify.com/drop` → glisser le fichier renommé `index.html` | 30 s |
| Vercel | `vercel deploy` dans un dossier contenant `index.html` | 1 min |
| GitHub Pages | Pousser `index.html`, activer Pages dans les réglages du dépôt | 3 min |
| Test local | `python3 -m http.server` puis tunnel `ngrok http 8000` | 2 min |

Renommer le fichier en `index.html` avant de le déposer.

## Ce que fait le prototype

- Caméra frontale, rendu miroir, suivi du visage en temps réel (MediaPipe Face Landmarker).
- La perruque suit la position, l'échelle et l'inclinaison de la tête, avec une
  compensation légère de la rotation gauche/droite.
- Quatre modèles, sélectionnés dans la barre du bas.
- Réglage manuel : sliders taille et hauteur, ou glisser directement sur l'image.
- Capture photo, enregistrement et partage natif sur mobile.
- Repli automatique : si le suivi ne se charge pas, la perruque reste positionnable
  à la main plutôt que de bloquer la démo.

## Limites assumées à ce stade

- **Pas de segmentation des cheveux.** La perruque se superpose aux cheveux existants
  au lieu de les masquer. Visible si l'utilisatrice a un volume important.
- **Pas de profondeur.** Les mèches passent devant les épaules quelle que soit la pose.
- **Rotation limitée.** Au-delà d'environ 30° de profil, le calage se dégrade.
- **Visuels reconstruits.** Les perruques sont composées à partir de la matière réelle
  de vos photos (couleur, grain, ondulation), montées dans une silhouette portable de
  face. Ce ne sont pas des détourages de vos photos produit, qui montrent les perruques
  tenues à la main, de côté ou de dos — inexploitables tels quels en superposition.

## Modifier le catalogue

Dans `essayage-perruques.html`, la liste en tête du `<script>` :

```js
const WIGS = [
  {id:'blonde',   name:'Blond 613',       detail:'Lisse · closure 4×4'},
  ...
];
```

Les noms et références sont des hypothèses tirées des photos : à corriger avec la
nomenclature réelle.

## Ajouter une perruque

`make_wigs.py` produit les assets. Pour un nouveau modèle :

1. Ajouter une entrée dans `STYLES` : chemin de la photo produit et `crop` cadrant
   une zone de **cheveux purs** (ni main, ni mur, ni table).
2. Régler `wave_amp` / `wave_period` (ondulation), `frizz` (mèches folles aux bords),
   `root_dark` (profondeur des racines), `length` et `taper`.
3. `python3 make_wigs.py`, puis réinjecter les `.webp` en base64 dans le HTML.

Le meilleur gain de qualité viendrait cependant de **photos produit de face, sur tête
mannequin, fond uni** : elles permettraient un détourage direct, plus fidèle que la
reconstruction actuelle.

## Calage du positionnement

Trois constantes en tête du script, si la perruque tombe trop haut ou trop bas :

```js
const ANCHOR_X = 500, ANCHOR_Y = 200;   // point d'ancrage dans l'image (1000×1500)
const WIDTH_RATIO = 2.62;               // largeur perruque ÷ largeur du visage
```

`ANCHOR_Y` plus grand fait descendre la perruque sur le front.
`WIDTH_RATIO` plus grand l'élargit.

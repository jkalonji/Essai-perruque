// Logique partagée par les pages du site (accueil + boutique) : chargement
// du catalogue de démo (assets/catalog.json), rendu des cartes produit, et
// petit toast de retour visuel. Catalogue en dur pour l'instant (pas de
// backend produit) -> a remplacer par un vrai catalogue si le site passe en
// boutique fonctionnelle un jour.

async function loadCatalog() {
  const res = await fetch("/assets/catalog.json");
  return res.json();
}

function productImageHTML(product) {
  if (product.image) {
    return `<img src="/${product.image}" alt="${product.name}" loading="lazy" />`;
  }
  // Pas encore de photo pour ce produit démo -> un aplat texturé distinct
  // par type plutôt qu'une fausse photo, pour rester honnête sur ce qui est
  // un vrai visuel (les 3 produits photographiés) et ce qui ne l'est pas.
  return `<div class="swatch swatch--${product.swatch}"><span>Aperçu à venir</span></div>`;
}

function productCardHTML(product) {
  return `
    <article class="product-card">
      <div class="product-card__media">${productImageHTML(product)}</div>
      <div class="product-card__body">
        <p class="eyebrow">${product.texture}</p>
        <h3>${product.name}</h3>
        <p class="product-card__desc">${product.desc}</p>
        <p class="product-card__price">${product.price} €</p>
        <div class="product-card__actions">
          <a class="btn" href="/essayer/?style=${product.essai_style}">Essayer virtuellement</a>
          <button class="btn primary" data-add-to-cart="${product.name}">Ajouter au panier</button>
        </div>
      </div>
    </article>
  `;
}

function showToast(message) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("visible"), 2600);
}

// Délégation sur tout le document : marche pour les cartes injectées côté
// accueil (teaser) et côté boutique sans dupliquer l'écouteur.
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-add-to-cart]");
  if (!btn) return;
  // Site vitrine (pas de vraie boutique, cf. décision produit) -> message
  // honnête plutôt qu'un panier qui ferait semblant de fonctionner.
  showToast(`"${btn.dataset.addToCart}" — boutique de démonstration, achat pas encore disponible.`);
});

"""
FLUX.1-Kontext-dev (edition d'image par instruction texte) en local.
Coiffure choisie via un argument en ligne de commande, ex :
    python run_kontext.py brushing_frange
(sans argument -> DEFAULT_STYLE). Voir HAIRSTYLES pour la liste des styles
disponibles et ajouter un nouveau style.

Le transformer (~12B params) est quantifie en FP8 poids-seuls (torchao) ->
Blackwell (RTX 50xx) a des tensor cores FP8 natifs, contrairement au NF4
bitsandbytes qui tournait dans un mode generique lent sur cette architecture
recente (constate : 100% GPU mais 41W/180W de conso, donc pas de calcul
tensor-core efficace).

Le premier lancement lit le transformer bf16 complet (~24 Go) et le
quantifie en FP8 en memoire -> c'est la partie lente du chargement (shards
bf16 + conversion). Ce transformer FP8 (~12 Go) est ensuite sauve une seule
fois dans TRANSFORMER_FP8_DIR ; les lancements suivants le rechargent tel
quel, sans repasser par le bf16 ni requantifier (les autres composants -
CLIP, T5, VAE - restent charges normalement depuis le cache HF, ils sont
petits/rapides et n'etaient pas le goulot d'etranglement).

Constate sur un premier run complet (28 steps) : ~4 min/step (1h50 au
total), avec GPU a 100% d'utilisation mais seulement ~35-40W/180W de
conso -> le GPU est en famine memoire, pas en calcul (meme symptome que le
probleme NF4 ci-dessus : la quantification FP8 poids-seuls ne compresse que
le stockage, le calcul repasse par un chemin generique). VRAM quasi pleine
(16 Go) + RAM systeme sous pression (pagefile sollicite) -> le driver
NVIDIA bascule probablement une partie des allocations GPU vers la RAM
systeme (voire le disque), ce qui explique le facteur de ralentissement.
D'ou 3 mitigations ci-dessous : image d'entree redimensionnee (moins
d'activations), moins de steps, et cpu offload (garde seulement le
composant actif sur le GPU) pour degager de la marge VRAM.
"""
import datetime
import os
import sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, TorchAoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import load_image
from torchao.quantization import Float8WeightOnlyConfig

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
FACE = "images/moi.jpg"
OUT_DIR = "results"
TRANSFORMER_FP8_DIR = "models/flux_kontext_transformer_fp8"
# Mapping fichier genere <-> prompt exact utilise, une ligne par run, lisible
# directement (cat/grep/less) : timestamp | style | fichier | LoRA | prompt.
# Sert a choisir une photo dans results/ sans avoir a rouvrir le log du run
# qui l'a produite. Chaque run ecrit son fichier de sortie avec un nom
# horodate (jamais ecrase) et ajoute une ligne ici.
GENERATION_LOG = f"{OUT_DIR}/generation_log.tsv"
MAX_SIDE = 1024  # redimensionne l'image d'entree si plus grande, pour limiter la VRAM prise par les activations
NUM_STEPS = 20  # 28 par defaut ; 20 suffit largement pour ce genre d'edition, quasi lineaire sur le temps total

# Clause de preservation d'identite, ajoutee automatiquement a TOUS les
# prompts (cf. main()), quel que soit le style. Le but du projet est que
# l'utilisateur se reconnaisse avec une coiffure differente, pas de generer
# une personne differente. Sans cette clause, un prompt qui decrit une
# coiffure via un vocabulaire fortement genre ("long", "face-framing",
# "sleek and neatly styled"...) peut faire deriver le modele vers un tout
# autre visage/genre (constate et reproduit 2 fois sur "brushing_frange"
# v3, corrige en v4 avec cette clause -> voir kontext_brushing_frange_run11_v4.png,
# le rendu le plus proche du cahier des charges obtenu jusqu'ici).
IDENTITY_GUARD = " Keep the same face and identity unchanged."

# Coiffures disponibles : cle -> (prompt envoye au modele, LoRA optionnel).
# Le LoRA "broccoli" est un fine-tuning specifique a cette forme -> il biaise
# le resultat vers cette texture meme si le prompt dit autre chose, donc on
# ne le charge que pour le style "broccoli". Les autres styles s'appuient
# uniquement sur le prompt, FLUX.1-Kontext-dev etant un editeur d'image
# generaliste par instruction (pas besoin de LoRA dedie par coiffure).
HAIRSTYLES = {
    "broccoli": (
        "Change hair to a broccoli haircut.",
        "fal/Broccoli-Hair-Kontext-Dev-LoRA",
    ),
    "brushing_frange": (
        # v1 "voluminous blowout (brushing) with a fringe" a donne des
        # dreadlocks pointues radiales -> "brushing" est un faux-ami anglais
        # (evoque une brosse a cheveux, pas un brushing/mise en forme) et
        # "voluminous" pousse vers une masse ronde gonflee ; reformule avec
        # du vocabulaire coiffure standard, sans le mot piege.
        # v2 "soft round-brush blowout with wispy bangs and gentle loose
        # waves" -> resultat proche d'une wolfcut (trop court/effile,
        # "wispy" pousse vers ebouriffe).
        # v3 "long, smooth blowout ... soft face-framing bangs ... sleek and
        # neatly styled" -> meilleure coiffure obtenue jusqu'ici, MAIS visage
        # transforme en femme (lunettes disparues, epaules denudees, bijoux
        # ajoutes), reproductible sur 2 runs.
        # v2 retestee seule : visage intact (lunettes, identite preservees),
        # mais coiffure repartie sur une afro herissee, pas le look voulu ->
        # le declencheur de la derive de genre est bien le vocabulaire
        # "long past the shoulders" + "face-framing" + "sleek and neatly
        # styled" de v3, pas juste le bonnet qui masque les vrais cheveux.
        # v4 : reprend la longueur/lisse de v3, retire "face-framing" ;
        # IDENTITY_GUARD (ajoutee automatiquement, cf. plus haut) fait le
        # reste -> meilleur resultat obtenu (visage/lunettes/barbe intacts).
        "Change hairstyle to a long, sleek blowout past the shoulders with "
        "soft bangs and gentle loose waves, neatly styled.",
        None,
    ),
    "brushing_frange_femme": (
        # Meme coiffure que v4, mais en assumant explicitement le genre
        # plutot que de demander de le preserver -> teste si le fait de
        # lever l'ambiguite (au lieu de lutter contre le biais du modele)
        # donne un edit plus propre/coherent. Resultat : "This is a woman"
        # l'emporte sur IDENTITY_GUARD, le visage change quand meme -> a
        # eviter si l'objectif est de se reconnaitre (garde pour reference).
        "This is a woman. Change her hairstyle to a long, sleek blowout "
        "past the shoulders with soft bangs and gentle loose waves, neatly "
        "styled.",
        None,
    ),
}
DEFAULT_STYLE = "brushing_frange"


def build_pipe(lora_repo=None):
    if os.path.isdir(TRANSFORMER_FP8_DIR):
        print("-> transformer FP8 deja quantifie trouve en local, chargement direct...")
        transformer = FluxTransformer2DModel.from_pretrained(
            TRANSFORMER_FP8_DIR, torch_dtype=torch.bfloat16,
        )
        pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16,
        )
    else:
        print("-> pas de cache FP8 local : quantification depuis le bf16 (lent, une seule fois)...")
        quant_config = PipelineQuantizationConfig(
            quant_mapping={"transformer": TorchAoConfig(quant_type=Float8WeightOnlyConfig())},
        )
        pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        os.makedirs(TRANSFORMER_FP8_DIR, exist_ok=True)
        pipe.transformer.save_pretrained(TRANSFORMER_FP8_DIR)
        print("-> transformer FP8 sauve dans", TRANSFORMER_FP8_DIR, "pour les prochains lancements")

    if lora_repo:
        pipe.load_lora_weights(lora_repo)
    # enable_model_cpu_offload() plutot que pipe.to("cuda") : ne garde sur le
    # GPU que le composant en train de travailler (transformer, VAE, ...) et
    # rapatrie le reste sur CPU entre les etapes -> plus de marge VRAM,
    # legerement plus lent par etape mais evite le fallback memoire
    # partagee/disque quand la VRAM est proche de la saturation.
    pipe.enable_model_cpu_offload()
    return pipe


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STYLE
    if style not in HAIRSTYLES:
        sys.exit(f"Coiffure inconnue: {style!r}. Choix possibles: {', '.join(HAIRSTYLES)}")
    prompt, lora_repo = HAIRSTYLES[style]
    prompt += IDENTITY_GUARD

    print(f"-> chargement du pipeline (transformer en FP8 torchao, coiffure={style})...")
    pipe = build_pipe(lora_repo=lora_repo)
    print("-> pipeline pret (cpu offload actif)")

    image = load_image(FACE)
    if max(image.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(image.size)
        new_size = (round(image.width * ratio), round(image.height * ratio))
        image = image.resize(new_size)
        print(f"-> image redimensionnee a {new_size} (limite VRAM)")

    # Seul le transformer Flux (boucle de denoising, NUM_STEPS appels) beneficie du
    # forcage EFFICIENT_ATTENTION. Le T5 (encodage du texte, 1 seul appel)
    # utilise un biais de position relatif incompatible avec ce backend -> on
    # le laisse sur le dispatch automatique (MATH), sans impact sur la vitesse
    # globale puisqu'il ne tourne qu'une fois.
    orig_forward = pipe.transformer.forward

    def patched_forward(*args, **kwargs):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return orig_forward(*args, **kwargs)

    pipe.transformer.forward = patched_forward

    print(f"-> prompt: {prompt!r}")
    print(f"-> generation ({NUM_STEPS} steps, EFFICIENT_ATTENTION force sur le transformer uniquement)...")
    result = pipe(
        image=image, prompt=prompt, guidance_scale=2.5, num_inference_steps=NUM_STEPS,
    ).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{OUT_DIR}/kontext_{style}_{timestamp}.png"
    result.save(out_path)
    print("->", out_path)

    log_line = "\t".join([timestamp, style, out_path, lora_repo or "-", prompt])
    is_new_log = not os.path.exists(GENERATION_LOG)
    with open(GENERATION_LOG, "a", encoding="utf-8") as f:
        if is_new_log:
            f.write("\t".join(["timestamp", "style", "fichier", "lora", "prompt"]) + "\n")
        f.write(log_line + "\n")
    print("-> logge dans", GENERATION_LOG)


if __name__ == "__main__":
    main()

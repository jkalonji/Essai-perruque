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
# Seed fixe pour le bruit initial -> meme entree (image+prompt+params) donne
# toujours la meme sortie. Sans ca, chaque run tire un bruit initial
# different et le resultat peut varier fortement (visage preserve ou non
# sur des prompts quasi identiques, cf. commentaires dans HAIRSTYLES) ce qui
# rend impossible de distinguer un effet du prompt d'un simple coup de
# chance. Change SEED pour explorer d'autres resultats a prompt fixe.
#
# Test de controle fait avec seed=0 sur brushing_frange_femme et curly_femme
# (memes prompts que sur des runs precedents en seed aleatoire) :
# - brushing_frange_femme (vocabulaire "blowout/face-framing/sleek") : derive
#   du visage a chaque fois, seed aleatoire ou seed=0 -> effet robuste du
#   vocabulaire, independant du seed.
# - curly_femme (vocabulaire "curly/ringlet") : visage preserve sur 2 runs a
#   seed aleatoire, mais derive (age y compris : "girl" rendu comme une
#   enfant) avec seed=0 -> donc PAS un effet purement deterministe du
#   vocabulaire dans ce cas, le seed a aussi son mot a dire. Conclusion
#   affinee : certains prompts ont un vocabulaire qui pousse la derive assez
#   fort pour l'emporter sur (presque) tous les seeds (cas blowout), d'autres
#   sont proches d'un seuil ou le seed peut faire basculer le resultat (cas
#   curly). A creuser plus tard : tester plusieurs seeds a prompt fixe sur
#   curly_femme pour estimer ce seuil, si la preservation d'identite devient
#   prioritaire sur ce style.
SEED = 0

# Note : on a teste une clause IDENTITY_GUARD ("Keep the same face and
# identity unchanged.") ajoutee automatiquement a tous les prompts, pour
# eviter que le modele ne derive vers un visage/genre different. Retiree :
# resultat pas fiable (parfois ignoree par le modele, cf. "brushing_frange_femme"
# et "curly_femme" -> comportement different sur des prompts quasi
# identiques, pas de seed fixee donc pas totalement reproductible). Le genre
# de la personne est desormais mentionne explicitement dans chaque prompt
# de HAIRSTYLES quand c'est pertinent, plutot que de compter sur une clause
# generique.

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
        # v4 : reprend la longueur/lisse de v3, retire "face-framing", genre
        # mentionne explicitement ("This is a man.") plutot que de compter
        # sur une clause de preservation d'identite generique (abandonnee,
        # cf. note plus haut).
        "This is a man. Change his hairstyle to a long, sleek blowout past "
        "the shoulders with soft bangs and gentle loose waves, neatly "
        "styled.",
        None,
    ),
    "brushing_frange_femme": (
        # Meme coiffure que v4, mais en genre feminin -> visage attendu
        # different de l'utilisateur (voulu ici).
        "This is a woman. Change her hairstyle to a long, sleek blowout "
        "past the shoulders with soft bangs and gentle loose waves, neatly "
        "styled.",
        None,
    ),
    "curly": (
        # Vocabulaire neutre/court, sans les mots ayant cause des derives
        # sur brushing_frange ("voluminous", "wispy", "face-framing").
        # "curly hair" seul est reparti sur une texture afro herissee -> pas
        # le look voulu (boucles definies, pas crepu). Genre mentionne
        # explicitement, comme brushing_frange v4.
        "This is a man. Change his hairstyle to natural curly hair, medium "
        "length, neatly styled.",
        None,
    ),
    "curly_femme": (
        # Demande explicitement une femme, visage different de l'utilisateur
        # attendu (voulu ici). "well-defined ringlet curls, not frizzy, not
        # afro-textured" pour eviter la derive vers l'afro herissee
        # constatee sur le style "curly".
        "This is a girl. Change her hairstyle to well-defined curly hair, "
        "spiral ringlet curls, not frizzy, not afro-textured, medium "
        "length, neatly styled.",
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

    generator = torch.Generator(device="cuda").manual_seed(SEED)

    print(f"-> prompt: {prompt!r}")
    print(f"-> generation ({NUM_STEPS} steps, seed={SEED}, EFFICIENT_ATTENTION force sur le transformer uniquement)...")
    result = pipe(
        image=image, prompt=prompt, guidance_scale=2.5, num_inference_steps=NUM_STEPS,
        generator=generator,
    ).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{OUT_DIR}/kontext_{style}_{timestamp}.png"
    result.save(out_path)
    print("->", out_path)

    log_line = "\t".join([timestamp, style, out_path, lora_repo or "-", f"seed={SEED}", prompt])
    is_new_log = not os.path.exists(GENERATION_LOG)
    with open(GENERATION_LOG, "a", encoding="utf-8") as f:
        if is_new_log:
            f.write("\t".join(["timestamp", "style", "fichier", "lora", "seed", "prompt"]) + "\n")
        f.write(log_line + "\n")
    print("-> logge dans", GENERATION_LOG)


if __name__ == "__main__":
    main()

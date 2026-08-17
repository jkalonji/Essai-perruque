"""
FLUX.1-Kontext-dev (edition d'image par instruction texte) en local, avec le
LoRA "broccoli hair" en bonus.

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
MAX_SIDE = 1024  # redimensionne l'image d'entree si plus grande, pour limiter la VRAM prise par les activations
NUM_STEPS = 20  # 28 par defaut ; 20 suffit largement pour ce genre d'edition, quasi lineaire sur le temps total


def build_pipe(with_lora=False):
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

    if with_lora:
        pipe.load_lora_weights("fal/Broccoli-Hair-Kontext-Dev-LoRA")
    # enable_model_cpu_offload() plutot que pipe.to("cuda") : ne garde sur le
    # GPU que le composant en train de travailler (transformer, VAE, ...) et
    # rapatrie le reste sur CPU entre les etapes -> plus de marge VRAM,
    # legerement plus lent par etape mais evite le fallback memoire
    # partagee/disque quand la VRAM est proche de la saturation.
    pipe.enable_model_cpu_offload()
    return pipe


def main():
    print("-> chargement du pipeline (transformer en FP8 torchao)...")
    pipe = build_pipe(with_lora=True)
    print("-> pipeline pret (cpu offload actif)")

    image = load_image(FACE)
    if max(image.size) > MAX_SIDE:
        ratio = MAX_SIDE / max(image.size)
        new_size = (round(image.width * ratio), round(image.height * ratio))
        image = image.resize(new_size)
        print(f"-> image redimensionnee a {new_size} (limite VRAM)")
    prompt = "Change hair to a broccoli haircut"

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

    print(f"-> generation ({NUM_STEPS} steps, EFFICIENT_ATTENTION force sur le transformer uniquement)...")
    result = pipe(
        image=image, prompt=prompt, guidance_scale=2.5, num_inference_steps=NUM_STEPS,
    ).images[0]
    out_path = f"{OUT_DIR}/kontext_broccoli_test.png"
    result.save(out_path)
    print("->", out_path)


if __name__ == "__main__":
    main()

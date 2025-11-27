import os
from datetime import datetime
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image
from PIL import ImageOps


def _prepare_output_path(output_dir: str, theme: str, suffix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_theme = theme.replace(" ", "_").replace("/", "-")
    filename = f"{timestamp}_{safe_theme}_{suffix}"
    return os.path.join(output_dir, filename)


def build_prompt(theme: str, composition: str) -> str:
    base = "monochrome rough sketch, clean lineart, pencil shading, 512x512"
    return f"{base}. theme: {theme}. layout: {composition}".replace("\n", " ")


def generate_image(
    *, theme: str, composition: str, config: dict[str, Any], output_dir: str
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    model_id = config.get("model_name", "stabilityai/sd-turbo")
    steps = int(config.get("num_inference_steps", 3))
    steps = max(1, min(6, steps))
    guidance_scale = float(config.get("guidance_scale", 1.8))
    negative_prompt = config.get(
        "negative_prompt",
        "low quality, blurry, distortions, watermark, text, color, nsfw",
    )
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed")
    lora_path = config.get("lora_path") if config.get("use_lora") else None

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )

    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if config.get("enable_xformers", True):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers が見つからないため通常の注意機構を使用します。")

    if config.get("attention_slicing", True):
        pipe.enable_attention_slicing()

    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name="lora")
        try:
            pipe.fuse_lora(lora_scale=1.0)
        except Exception:
            pass

    prompt = build_prompt(theme, composition)
    generator = None
    if isinstance(seed, int):
        generator = torch.Generator(device=device).manual_seed(seed)

    with torch.autocast(device_type=device, dtype=torch_dtype):
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            generator=generator,
        )
    image = result.images[0]
    grayscale = ImageOps.grayscale(image)

    image_path = _prepare_output_path(output_dir, theme, "rough.png")
    grayscale.save(image_path)

    prompt_log = _prepare_output_path(output_dir, theme, "prompt.txt")
    with open(prompt_log, "w", encoding="utf-8") as f:
        f.write("# positive\n")
        f.write(prompt + "\n")
        f.write("\n# negative\n")
        f.write(negative_prompt + "\n")

    return image_path, prompt_log


__all__ = ["generate_image", "build_prompt"]

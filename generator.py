import logging
import os
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image, OnnxStableDiffusionXLPipeline
from PIL import ImageOps



logger = logging.getLogger(__name__)
 main


def _prepare_output_path(output_dir: str, theme: str, suffix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_theme = theme.replace(" ", "_").replace("/", "-")
    filename = f"{timestamp}_{safe_theme}_{suffix}"
    return os.path.join(output_dir, filename)


def _save_grayscale_image(image, path: Path) -> Path:
    ext = path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("出力ファイルは PNG または JPEG を指定してください。")

    path.parent.mkdir(parents=True, exist_ok=True)
    grayscale = ImageOps.grayscale(image)
    image_format = "PNG" if ext == ".png" else "JPEG"
    grayscale.save(path, format=image_format)
    return path


def build_prompt(theme: str, composition: str) -> str:
    base = "monochrome rough sketch, clean lineart, pencil shading, 512x512"
    return f"{base}. theme: {theme}. layout: {composition}".replace("\n", " ")


def generate_image(
    *, theme: str, composition: str, config: dict[str, Any], output_dir: str
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    runtime = _detect_environment(config)
    model_id = config.get("model_name", "stabilityai/sd-turbo")
    requested_steps = int(config.get("num_inference_steps", 3))
    steps = _inference_steps(runtime, requested_steps)
    guidance_scale = float(config.get("guidance_scale", 1.8))
    negative_prompt = config.get(
        "negative_prompt",
        "low quality, blurry, distortions, watermark, text, color, nsfw",
    )
    device = "cuda" if runtime["mode"].startswith("torch") else "cpu"
    seed = config.get("seed")
    lora_path = config.get("lora_path") if config.get("use_lora") else None


    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = _initialize_pipeline(
        model_id=model_id,
        torch_dtype=torch_dtype,
        device=device,
        enable_xformers=config.get("enable_xformers", True),
        attention_slicing=config.get("attention_slicing", True),
        lora_path=lora_path,
    )

    prompt = build_prompt(theme, composition)
    generator = None
    if isinstance(seed, int):
        generator = torch.Generator(device=device).manual_seed(seed)

    autocast_context = (
        torch.autocast(device_type=device, dtype=torch_dtype)
        if device == "cuda"
        else nullcontext()
    )

    try:
        with autocast_context:
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
        image_path = Path(_prepare_output_path(output_dir, theme, "rough.png"))
        saved_image_path = _save_grayscale_image(image, image_path)

        prompt_log = _prepare_output_path(output_dir, theme, "prompt.txt")
        with open(prompt_log, "w", encoding="utf-8") as f:
            f.write("# positive\n")
            f.write(prompt + "\n")
            f.write("\n# negative\n")
            f.write(negative_prompt + "\n")

        return str(saved_image_path), prompt_log
    except Exception as exc:
        logger.exception("Failed to generate image: %s", exc)
        raise RuntimeError("画像の生成に失敗しました。") from exc


def _initialize_pipeline(
    *,
    model_id: str,
    torch_dtype: torch.dtype,
    device: str,
    enable_xformers: bool = True,
    attention_slicing: bool = True,
    lora_path: str | None = None,
):
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )
    except Exception as exc:
        logger.exception("Failed to load pipeline: %s", exc)
        raise RuntimeError("画像生成パイプラインの読み込みに失敗しました。") from exc

    pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.warning(
                "xformers が見つからないため通常の注意機構を使用します: %s", exc
            )

    if attention_slicing:
        pipe.enable_attention_slicing()

    if lora_path:
        try:
            pipe.load_lora_weights(lora_path, adapter_name="lora")
            pipe.fuse_lora(lora_scale=1.0)
        except Exception as exc:
            logger.warning("LoRA のロードに失敗しました: %s", exc)
 main

    return pipe


 main

def generate_sketch(text: str, output_path: Path) -> Path:
    """Generate a monochrome sketch image and save to the given path."""

    if not text or not text.strip():
        raise ValueError("プロンプトが空です。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = _initialize_pipeline(
        model_id="stabilityai/sd-turbo",
        torch_dtype=torch_dtype,
        device=device,
        enable_xformers=True,
        attention_slicing=True,
        lora_path=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    autocast_context = (
        torch.autocast(device_type=device, dtype=torch_dtype)
        if device == "cuda"
        else nullcontext()
    )

    try:
        with autocast_context:
            result = pipe(
                text,
                num_inference_steps=3,
                guidance_scale=1.8,
                height=512,
                width=512,
            )
        image = result.images[0]
        _save_grayscale_image(image, output_path)
        logger.info("生成したスケッチを保存しました: %s", output_path)
        return output_path
    except Exception as exc:
        logger.exception("Failed to generate sketch: %s", exc)
        raise RuntimeError("スケッチ画像の生成に失敗しました。") from exc


__all__ = ["generate_image", "generate_sketch", "build_prompt"]

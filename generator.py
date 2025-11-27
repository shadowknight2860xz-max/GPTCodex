import os
from datetime import datetime
from typing import Any

import torch
from diffusers import AutoPipelineForText2Image, OnnxStableDiffusionXLPipeline
from PIL import ImageOps


def _detect_environment(config: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "requested_device": config.get("device"),
        "cuda_available": torch.cuda.is_available(),
        "total_vram_gb": None,
        "mode": "onnx",
        "providers": ["CPUExecutionProvider"],
    }

    if runtime["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            runtime["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
        except Exception:
            runtime["total_vram_gb"] = None

    forced_cpu = runtime["requested_device"] == "cpu"
    prefer_cuda = runtime["cuda_available"] and not forced_cpu

    if prefer_cuda:
        vram = runtime["total_vram_gb"] or 0
        if vram and vram < 3:
            runtime["mode"] = "onnx"
        elif vram <= 4:
            runtime["mode"] = "torch_low_vram"
        else:
            runtime["mode"] = "torch"
    else:
        runtime["mode"] = "onnx"

    if runtime["mode"] == "onnx" and runtime["cuda_available"]:
        runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("[runtime] cuda_available:", runtime["cuda_available"])
    print("[runtime] requested_device:", runtime["requested_device"] or "auto")
    if runtime["total_vram_gb"] is not None:
        print("[runtime] total_vram_gb:", runtime["total_vram_gb"])
    print("[runtime] selected_mode:", runtime["mode"])
    print("[runtime] onnx_providers:", runtime["providers"])

    return runtime


def _inference_shape(runtime: dict[str, Any]) -> tuple[int, int]:
    if runtime["mode"] == "onnx":
        return 448, 448
    return 512, 512


def _inference_steps(runtime: dict[str, Any], requested_steps: int) -> int:
    if runtime["mode"] == "onnx":
        return max(1, min(4, requested_steps))
    if runtime["mode"] == "torch_low_vram":
        return max(1, min(5, requested_steps))
    return max(1, min(6, requested_steps))


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
    height, width = _inference_shape(runtime)

    prompt = build_prompt(theme, composition)
    generator = None
    if isinstance(seed, int):
        generator = torch.Generator(device=device).manual_seed(seed)

    if runtime["mode"].startswith("torch"):
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

        if runtime["mode"] == "torch_low_vram":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                print("sequential offload を有効化できませんでしたが続行します。")
        elif config.get("attention_slicing", True):
            pipe.enable_attention_slicing()

        if lora_path:
            pipe.load_lora_weights(lora_path, adapter_name="lora")
            try:
                pipe.fuse_lora(lora_scale=1.0)
            except Exception:
                pass

        print(
            f"[runtime] diffusers pipeline (device={device}, dtype={torch_dtype}, steps={steps}, size={height}x{width})"
        )
        with torch.autocast(device_type=device, dtype=torch_dtype):
            result = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
    else:
        providers = runtime.get("providers", ["CPUExecutionProvider"])
        pipe = OnnxStableDiffusionXLPipeline.from_pretrained(
            model_id,
            provider=providers[0],
            providers=providers,
            use_safetensors=True,
        )

        print(
            f"[runtime] onnxruntime pipeline (providers={providers}, steps={steps}, size={height}x{width})"
        )
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
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

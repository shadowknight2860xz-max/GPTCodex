import argparse
import os
from datetime import datetime

from generator import generate_sketch, _sanitize_filename


def _build_default_output_path(prompt: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = _sanitize_filename(prompt)
    return os.path.join("output", f"{timestamp}_{safe_name}_sketch.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="ラフスケッチ生成 CLI")
    parser.add_argument(
        "--prompt",
        "-p",
        help="生成に使うプロンプト。未指定時は対話で入力。",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="保存先ファイルパス (PNG)。未指定で output/ 配下に自動保存。",
    )
    parser.add_argument("--model", "-m", help="使用するモデル ID", default=None)
    parser.add_argument("--device", "-d", help="使用デバイス (cuda/cpu など)")
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="拡散のステップ数 (1-6)。省メモリ環境では 1-2 を推奨。",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.8,
        help="ガイダンススケール。SD Turbo は 0-2 付近が推奨。",
    )
    args = parser.parse_args()

    prompt = args.prompt or input("プロンプトを入力してください: ").strip()
    if not prompt:
        raise SystemExit("プロンプトが入力されていません。")

    output_path = args.output or _build_default_output_path(prompt)

    result_path = generate_sketch(
        prompt=prompt,
        output_path=output_path,
        model=args.model,
        device=args.device,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )

    print("生成完了")
    print(f"プロンプト: {prompt}")
    print(f"保存先: {result_path}")


if __name__ == "__main__":
    main()

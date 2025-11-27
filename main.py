import argparse
import json
import os
from datetime import datetime

from compose import generate_composition
from generator import generate_image


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(content: str, output_dir: str, theme: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_theme = theme.replace(" ", "_").replace("/", "-")
    path = os.path.join(output_dir, f"{timestamp}_{safe_theme}_composition.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="構図提案つきネーム生成AI (CLI)")
    parser.add_argument("theme", nargs="?", help="生成したいテーマ。未指定時は対話で入力。")
    parser.add_argument(
        "--config", default="config.json", help="設定ファイルのパス (JSON)"
    )
    args = parser.parse_args()

    if args.theme:
        theme = args.theme
    else:
        theme = input("テーマを入力してください: ").strip()
        if not theme:
            raise SystemExit("テーマが入力されていません。")

    config = load_config(args.config)
    output_dir = config.get("output_dir", "output")

    composition = generate_composition(theme, seed=config.get("seed"))
    text_path = save_text(composition, output_dir, theme)

    image_path, prompt_path = generate_image(
        theme=theme, composition=composition, config=config, output_dir=output_dir
    )

    print("生成完了")
    print(f"構図案: {text_path}")
    print(f"画像: {image_path}")
    print(f"プロンプトログ: {prompt_path}")


if __name__ == "__main__":
    main()

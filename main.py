import argparse
import json
from datetime import datetime
from pathlib import Path

from compose import generate_composition
from generator import build_output_path, generate_image


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(
    content: str,
    output_dir: Path | str,
    theme: str,
    *,
    overwrite: bool = False,
    timestamp: str | None = None,
) -> str:
    path = build_output_path(
        output_dir,
        theme,
        "composition.txt",
        overwrite=overwrite,
        timestamp=timestamp,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="構図提案つきネーム生成AI (CLI)")
    parser.add_argument("theme", nargs="?", help="生成したいテーマ。未指定時は対話で入力。")
    parser.add_argument(
        "--config", default="config.json", help="設定ファイルのパス (JSON)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="同名ファイルが存在する場合に上書き保存します",
    )
    args = parser.parse_args()

    if args.theme:
        theme = args.theme
    else:
        theme = input("テーマを入力してください: ").strip()
        if not theme:
            raise SystemExit("テーマが入力されていません。")

    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "output"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    composition = generate_composition(theme, seed=config.get("seed"))
    text_path = save_text(
        composition,
        output_dir,
        theme,
        overwrite=args.overwrite,
        timestamp=timestamp,
    )

    image_path, prompt_path = generate_image(
        theme=theme,
        composition=composition,
        config=config,
        output_dir=output_dir,
        overwrite=args.overwrite,
        timestamp=timestamp,
    )

    print("生成完了")
    print(f"構図案: {text_path}")
    print(f"画像: {image_path}")
    print(f"プロンプトログ: {prompt_path}")


if __name__ == "__main__":
    main()

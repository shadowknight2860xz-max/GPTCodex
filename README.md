# 構図提案つきネーム生成AI（CLI版）

軽量な CLI で「構図案の文章生成」と「白黒ラフ画像生成」を同時に行います。GTX1650（VRAM 4GB）でも動作するよう、SD Turbo あるいは SD1.5 + LoRA の簡易構成を想定しています。

## 機能
- テーマを入力すると、カメラ・構図・ポーズ・光源・ネーム案を含む構図案を生成
- diffusers を使った 512x512 の白黒ラフ画像生成（ステップ数 1〜6）
- プロンプトログと結果ファイルを `output/` にまとめて保存
- config.json でモデルや LoRA 有無を切り替え可能
- xformers による省メモリ化（利用可能な環境で自動有効化）

## 事前準備
1. **Python のインストール**
   - Windows: [公式サイト](https://www.python.org/downloads/) から 3.10 以上を推奨
   - WSL/ Linux: ディストリビューションのパッケージマネージャで 3.10 以上を用意
2. **CUDA 版 PyTorch のインストール**
   - [PyTorch 公式のインストールコマンド](https://pytorch.org/get-started/locally/)を参照し、CUDA 対応版をインストールしてください。
   - 例: `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`
3. **リポジトリの取得**
   - 本ディレクトリに必要ファイルが揃っています。`output/` フォルダが無い場合は自動で作成されます。

## 依存関係のインストール
PyTorch をインストール済みであることを確認したうえで、以下を実行します。

```bash
pip install -r requirements.txt
```

## config.json の設定
`config.json` でモデルや推論設定を変更できます。

```json
{
  "model_name": "stabilityai/sd-turbo",   // SD Turbo で高速軽量
  "use_lora": false,                       // true にすると LoRA をロード
  "lora_path": "",                        // ローカルの LoRA safetensors へのパス
  "num_inference_steps": 3,                // 1〜6 推奨
  "guidance_scale": 1.8,                   // Turbo の場合は 0〜2 がおすすめ
  "negative_prompt": "low quality, blurry, distortions, watermark, text, color, nsfw",
  "enable_xformers": true,                 // xformers があれば自動で使用
  "attention_slicing": true,               // 省メモリのためのスライス
  "seed": 1234,                            // 再現性が欲しい場合に指定
  "output_dir": "output"                  // 出力先フォルダ
}
```

### モデルの入手方法
- **SD Turbo**: `stabilityai/sd-turbo` を自動ダウンロードします。512x512/低ステップ向きで軽量。
- **SD1.5 + LoRA**: `model_name` を `runwayml/stable-diffusion-v1-5` に変更し、`use_lora=true` と `lora_path` に LoRA ファイルのパスを指定してください（ローカル配置推奨）。

## 使い方（CLI）
### 直接実行
```bash
python main.py --prompt "夜の街を走るスケーター"
```
`--prompt` を省略すると対話的に入力できます。出力先を省略すると `output/` 配下に
`<タイムスタンプ>_<プロンプト一部>_sketch.png` の形式で自動保存します。

主なオプション:

| オプション | 説明 |
| --- | --- |
| `--prompt` / `-p` | 生成に使うプロンプト。未指定時は対話入力。 |
| `--output` / `-o` | 保存先ファイルパス (PNG)。未指定で自動命名。 |
| `--model` / `-m` | 使用するモデル ID。デフォルトは `stabilityai/sd-turbo`。 |
| `--device` / `-d` | `cuda`/`cpu` など使用デバイスを固定したい場合に指定。 |
| `--steps` | 拡散のステップ数 (1-6)。 |
| `--guidance` | ガイダンススケール。SD Turbo は 0-2 付近が推奨。 |

### start.bat（Windows 一発起動）
`start.bat` をダブルクリックすると、仮想環境 `.venv` を自動作成し、依存関係をインストールした上で `main.py` を実行します。引数を付ける場合はショートカットの「リンク先」に追記してください。

例:
```
start.bat --prompt "雨上がりの路地を歩く少年"
```

## 出力物
`output/` フォルダに以下が保存されます（タイムスタンプ＋プロンプトを含む自動命名）。
- 白黒ラフ画像 (PNG): `*_sketch.png`

## VRAM が足りない場合
- `num_inference_steps` を 1〜2 に下げる
- `guidance_scale` を 1.0 付近に下げる
- LoRA を使う場合、解凍済みのローカルファイルを指定して読み込み時間を短縮
- CPU 実行も可能ですが大幅に遅くなります。CUDA ドライバと xformers が有効な環境を推奨

## ファイル構成
- `main.py`: CLI 本体。プロンプト受け取りと画像生成の呼び出し
- `compose.py`: （参考）テーマから構図案テキストを作成するサンプル
- `generator.py`: diffusers で 512x512 の白黒ラフ画像を生成。xformers/LoRA対応
- `config.json`: 設定ファイル
- `requirements.txt`: 依存関係
- `start.bat`: Windows ワンクリック起動用
- `output/`: 生成物の保存先（無ければ自動作成）

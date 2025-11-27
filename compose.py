import random
from textwrap import dedent


CAMERA_ANGLES = [
    "俯瞰、35mm レンズ、軽い歪み",
    "アイレベル、50mm レンズ",
    "ローアングル、広角 28mm",
    "中望遠 85mm、柔らかいボケ",
]

COMPOSITIONS = [
    "三分割法で主役を左下に配置",
    "シンメトリー構図で中央に抜けを作る",
    "対角線構図で奥行きを強調",
    "トンネル構図で視線を導く",
]

POSES = [
    "腕を組んでカメラ目線",
    "歩きながら振り返る",
    "ジャンプの瞬間を切り取る",
    "椅子に腰掛けてリラックス",
]

LIGHTS = [
    "キーライトを左 45 度、リムライト弱",
    "逆光＋リムライトでシルエット強調",
    "柔らかい窓光、影は淡く",
    "スポットライトで主役だけを照らす",
]

NAME_IDEAS = [
    "静寂を破る瞬間",
    "風を抱く視線",
    "影と光の境界",
    "歩幅のリズム",
]


def _choose(items: list[str], rng: random.Random) -> str:
    return rng.choice(items)


def generate_composition(theme: str, seed: int | None = None) -> str:
    """テーマから構図案テキストを作成する。"""

    rng = random.Random(seed)
    camera = _choose(CAMERA_ANGLES, rng)
    composition = _choose(COMPOSITIONS, rng)
    pose = _choose(POSES, rng)
    light = _choose(LIGHTS, rng)
    name = _choose(NAME_IDEAS, rng)

    body = dedent(
        f"""
        - カメラ：{camera}
        - 構図：{composition}
        - ポーズ：{pose}
        - 光源：{light}
        - ネーム案：{name}
        """
    ).strip()

    return f"テーマ：{theme}\n" + body + "\n"


__all__ = ["generate_composition"]

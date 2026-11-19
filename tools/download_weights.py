"""
OSNet の Re-ID 用重みをダウンロードして osnet_weights/ に置くスクリプト

torchreid（deep-person-reid）公式 Model Zoo の Google Drive から gdown で取得し、
reid.py が読む名前（osnet_x1_0_msmt17.pth）で保存する。
SHA256 を確認するので、どのマシンでも同じ重みを使っていることを保証できる。

入手元: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
  「Same-domain ReID」表の osnet_x1_0 行・msmt17 列の model リンク

使い方（リポジトリ直下から。引数は不要）:
    python tools/download_weights.py
"""

import hashlib
import os
import shutil
import sys
import tempfile

# ============================================================
# 設定（ここを書き換えて実行する）
# ============================================================

# Google Drive のファイルID（Model Zoo のリンク https://drive.google.com/file/d/<ここ>/view の部分）
FILE_ID = "112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"

# 保存先（reid.py の MODEL_PATH と同じ）
OUTPUT_PATH = "osnet_weights/osnet_x1_0_msmt17.pth"

# 期待する SHA256（deca で使っている重みの値。2026-11-19 に sha256sum で確認）
# None にすると確認せずに保存し、計算した値を表示する
EXPECTED_SHA256 = "b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2"

# ============================================================


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_loadable(path):
    """PyTorch で読み込めるか（壊れたファイルや HTML のエラーページでないか）を確認する"""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    n = len(state) if hasattr(state, "__len__") else 0
    print(f"  読み込み確認: OK（パラメータのキー {n} 個）")


def main():
    if os.path.exists(OUTPUT_PATH):
        digest = sha256_of(OUTPUT_PATH)
        if EXPECTED_SHA256 is None or digest == EXPECTED_SHA256:
            print(f"すでにあります: {OUTPUT_PATH}\n  SHA256: {digest}")
            return
        print(f"⚠️ 既存のファイルの SHA256 が違うので取り直します\n  既存: {digest}\n  期待: {EXPECTED_SHA256}")

    import gdown

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = os.path.join(tmp, "weights.pth")
        print(f"ダウンロード中: Google Drive id={FILE_ID}")
        if gdown.download(id=FILE_ID, output=tmp_path, quiet=False) is None:
            sys.exit("❌ ダウンロードに失敗しました（リンク切れ・アクセス制限の可能性）")

        digest = sha256_of(tmp_path)
        print(f"  SHA256: {digest}")
        if EXPECTED_SHA256 is not None and digest != EXPECTED_SHA256:
            sys.exit(f"❌ SHA256 が一致しません。保存しませんでした\n  期待: {EXPECTED_SHA256}")
        check_loadable(tmp_path)
        shutil.move(tmp_path, OUTPUT_PATH)

    print(f"✅ 保存しました: {OUTPUT_PATH}")
    if EXPECTED_SHA256 is None:
        print("  ※ EXPECTED_SHA256 が未設定です。deca の重みの SHA256 と上の値が同じか確認してください")


if __name__ == "__main__":
    main()

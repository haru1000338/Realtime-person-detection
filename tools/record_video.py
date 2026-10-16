#!/usr/bin/env python3
"""検証用の生映像を録画する。推論は行わない。

下の設定を書き換えて `python tools/record_video.py` を実行する。停止は Ctrl+C。
画面のない環境を想定し、冒頭の1フレームだけ画角確認用に保存する。
"""

import os
import time
from datetime import datetime

import cv2

# =========================== 設定 ===========================

SOURCE = 0                      # USBなら 0 や "/dev/video0"、ネットワークなら "rtsp://..."
OUT_DIR = "tests/_recordings"
WIDTH, HEIGHT = 1280, 720
FPS = 30
MAX_SECONDS = 0                 # 0 なら Ctrl+C まで録り続ける

# ============================================================


def main():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"カメラを開けません: {SOURCE}")

    # MJPG を先に指定する。非圧縮(YUYV)だとUSB帯域が足りず数fpsしか出ない
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # 要求が通るとは限らないので実際の値を取り直す
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"{stamp}.mp4")

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"書き込み先を開けません: {path}")

    print(f"録画開始: {path}")
    print(f"  {width}x{height} / {fps:.1f} fps / {fourcc}")
    if fourcc != "MJPG":
        print("  ⚠️ MJPG が通らなかった。fps が出ない可能性がある")

    start = time.time()
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("フレームを取得できません。終了します。")
                break
            writer.write(frame)
            if count == 0:
                # 画面がないので、画角確認用に最初の1枚だけ残す
                cv2.imwrite(os.path.join(OUT_DIR, f"{stamp}_first.jpg"), frame)
            count += 1
            if MAX_SECONDS and time.time() - start >= MAX_SECONDS:
                break
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start
        cap.release()
        writer.release()

    actual = count / elapsed if elapsed else 0
    print(f"\n録画終了: {path}（{os.path.getsize(path) / 1e6:.1f} MB）")
    print(f"  {count}フレーム / {elapsed:.1f}秒 / 実測 {actual:.1f} fps（設定 {fps:.1f}）")

    # 実測と設定がずれていると再生速度が実時間と合わず、映像から滞在時間を測れない
    if fps and abs(actual - fps) / fps > 0.1:
        print(f"  ⚠️ 設定とずれている。FPS を {actual:.0f} にして録り直すこと")


if __name__ == "__main__":
    main()
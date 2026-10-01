#!/usr/bin/env python3
"""検証用の生映像を録画する。推論は一切行わない。

Dockerコンテナ内で実行する前提。SSH越しで画面がないため、
プレビューは出さず、代わりに一定間隔で静止画を保存して画角を確認する。

前提条件（カメラ接続後に確認すること）:
  1. ホストで `ls -l /dev/video*` にデバイスが現れていること
  2. docker-compose.yml に下記を追記し、コンテナを作り直していること
         devices:
           - /dev/video0:/dev/video0
  3. コンテナ内で以下が True を返すこと
         python -c "import cv2; c=cv2.VideoCapture(0); print(c.isOpened())"

使い方:
  下の「設定」を書き換えて `python tools/record_video.py` を実行する。
  停止は Ctrl+C。
"""

import csv
import os
import time
from datetime import datetime

import cv2

# =========================== 設定（ここだけ書き換える）===========================

# カメラ。整数（0, 1, ...）でも "/dev/video0" のようなパスでもよい
SOURCE = 0

# 保存先。tests/ 配下なので git では追跡されない
OUT_DIR = "tests/_recordings"

# 希望する解像度とフレームレート。カメラが対応していなければ無視される
WIDTH, HEIGHT = 1280, 720
FPS = 30

# MJPG で取得するか。
# USBカメラの多くは非圧縮(YUYV)だと帯域が足りず、1080pで5fps程度しか出ない。
# MJPG に対応していれば30fpsが出るため、既定で有効にしている。
# v4l2-ctl --list-formats-ext で対応状況を確認できる。
USE_MJPG = True

# 最大録画時間（秒）。0 なら Ctrl+C で止めるまで録り続ける
MAX_SECONDS = 0

# 何秒おきに確認用の静止画を保存するか。0 で無効。
# 画面が無いので、これで画角とピントを確認する
SNAPSHOT_EVERY = 10

# ==============================================================================


def fourcc_name(value):
    """VideoCapture が返す数値の FOURCC を 'MJPG' のような文字列に変換する"""
    v = int(value)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def main():
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise SystemExit(
            f"カメラを開けません: {SOURCE}\n"
            f"  - ホストで /dev/video* が存在するか\n"
            f"  - docker-compose.yml の devices にデバイスを追加したか\n"
            f"  - コンテナを作り直したか\n"
            f"を確認してください。"
        )

    # FOURCC は解像度より先に設定する。順序を逆にすると反映されないことがある
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # 要求した値がそのまま通るとは限らないため、実際の値を取り直す
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    fourcc = fourcc_name(cap.get(cv2.CAP_PROP_FOURCC))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    video_path = os.path.join(OUT_DIR, f"{stamp}.mp4")
    log_path = os.path.join(OUT_DIR, f"{stamp}_frames.csv")
    snap_dir = os.path.join(OUT_DIR, f"{stamp}_snapshots")

    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"書き込み先を開けません: {video_path}")

    if SNAPSHOT_EVERY:
        os.makedirs(snap_dir, exist_ok=True)

    print(f"録画開始: {video_path}")
    print(f"  解像度 {width}x{height} / {fps:.1f} fps / 形式 {fourcc}")
    if USE_MJPG and fourcc != "MJPG":
        print(f"  ⚠️ MJPG を要求したが {fourcc} になっている。fpsが出ない可能性がある")
    if SNAPSHOT_EVERY:
        print(f"  確認用の静止画を {SNAPSHOT_EVERY} 秒おきに {snap_dir} へ保存する")
    print("  停止: Ctrl+C")

    # フレームごとの実時刻を残す。
    # 映像だけでは「このフレームが何時何分か」が分からず、
    # 撮影時の手書きログと突き合わせられないため。
    log = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log)
    log_writer.writerow(["frame", "elapsed_sec", "unix_time"])

    start = time.time()
    count = 0
    next_snap = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("フレームを取得できませんでした。終了します。")
                break

            writer.write(frame)
            now = time.time()
            elapsed = now - start
            log_writer.writerow([count, f"{elapsed:.3f}", f"{now:.3f}"])
            count += 1

            if SNAPSHOT_EVERY and elapsed >= next_snap:
                cv2.imwrite(os.path.join(snap_dir, f"{int(elapsed):05d}s.jpg"), frame)
                next_snap += SNAPSHOT_EVERY

            if count % 300 == 0:
                print(f"  録画中 {elapsed:.0f}秒 / {count}フレーム "
                      f"（実測 {count / elapsed:.1f} fps）")

            if MAX_SECONDS and elapsed >= MAX_SECONDS:
                break
    except KeyboardInterrupt:
        print("\n停止しました。")
    finally:
        elapsed = time.time() - start
        cap.release()
        writer.release()
        log.close()

    size_mb = os.path.getsize(video_path) / 1024 / 1024
    actual_fps = count / elapsed if elapsed else 0
    print(f"\n録画終了")
    print(f"  ファイル  : {video_path}（{size_mb:.1f} MB）")
    print(f"  フレーム数: {count}")
    print(f"  録画時間  : {elapsed:.1f} 秒")
    print(f"  実測fps   : {actual_fps:.1f}（設定 {fps:.1f}）")
    print(f"  フレーム記録: {log_path}")

    # 実測fpsが設定と大きくずれていると再生速度が実時間と一致せず、
    # 映像から滞在時間を測ったときに誤差になる
    if fps and abs(actual_fps - fps) / fps > 0.1:
        print(f"\n⚠️ 実測fpsが設定と10%以上ずれている。")
        print(f"   このまま再生すると速度が実時間と一致しない。")
        print(f"   設定の FPS を {actual_fps:.0f} にして録り直すと正確になる。")


if __name__ == "__main__":
    main()
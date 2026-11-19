"""
長い録画の「人が多い時間帯」を探すスクリプト

録画を一定間隔（SAMPLE_EVERY_SEC 秒ごと）に1フレームだけ取り出して YOLO で人数を数え、
人数の時系列（CSV と グラフ）と、人数が多い区間の上位を出力する。
追跡や Re-ID はしないので、1.5時間の録画でも数分で終わる見込み。
ここで見つけた区間を eval_tracking.py の START_SECONDS / MAX_SECONDS に設定して使う。

使い方:
  1. 下の「設定」の定数を書き換える（引数は不要）
  2. リポジトリ直下から実行する
       CUDA_VISIBLE_DEVICES=0 python tools/scan_people.py

出力（OUTPUT_DIR 以下）:
  - <動画名>__people__<日付>.csv  … 秒数ごとの人数
  - people-timeline__<日付>.png   … 全動画の人数の推移（動画ごとに1段）
  - ターミナルに、各動画で人数が多い区間の上位 TOP_K 件（開始秒・平均人数）
"""

import csv
import os
from datetime import date

import cv2
import numpy as np

# ============================================================
# 設定（ここを書き換えて実行する）
# ============================================================

VIDEO_PATHS = [
    "tests/_fsm01-split/part_002.avi",
    "tests/_fsm01-split/part_003.avi",
    "tests/_fsm01-split/part_004.avi",
]

OUTPUT_DIR = "tests/_fsm01-split-scan"

# 本番と同じ重み
YOLO_WEIGHTS = "yolo26s.pt"

# 人として数える検出スコアの下限（ByteTrack が新しいトラックを作る new_track_thresh と同じ値）
CONF = 0.6

# 何秒ごとに1フレームを数えるか
SAMPLE_EVERY_SEC = 1.0

# 「人が多い区間」を探すときの区間の長さ（秒）と、表示する件数
WINDOW_SEC = 600
TOP_K = 3

# ============================================================
# ここから下は書き換えなくてよい
# ============================================================

TODAY = date.today().isoformat()


def scan(model, path):
    """(秒数の配列, 人数の配列, fps) を返す"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(SAMPLE_EVERY_SEC * fps)))
    times, counts = [], []
    idx = 0
    while True:
        # 数えないフレームは grab() で読み飛ばす（画像に展開しないので速い）
        if idx % step != 0:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        boxes = model.predict(frame, conf=CONF, classes=[0], verbose=False)[0].boxes
        times.append(idx / fps)
        counts.append(len(boxes))
        if len(times) % 600 == 0:
            print(f"  {os.path.basename(path)}: {idx / fps / 60:.1f} 分まで処理")
        idx += 1
    cap.release()
    return np.array(times), np.array(counts), fps


def top_windows(times, counts):
    """WINDOW_SEC 秒の区間ごとの平均人数を出し、多い順に TOP_K 件（重ならないように）返す"""
    n = max(1, int(WINDOW_SEC / SAMPLE_EVERY_SEC))
    if len(counts) < n:
        return [(0.0, float(counts.mean()) if len(counts) else 0.0)]
    means = np.convolve(counts, np.ones(n) / n, mode="valid")
    picked = []
    for i in np.argsort(means)[::-1]:
        if all(abs(i - j) >= n for j, _ in picked):
            picked.append((i, means[i]))
        if len(picked) >= TOP_K:
            break
    return [(float(times[i]), float(m)) for i, m in picked]


def main():
    from ultralytics import YOLO

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(YOLO_WEIGHTS)
    results = []
    for path in VIDEO_PATHS:
        print(f"{path} を処理中...")
        times, counts, fps = scan(model, path)
        name = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(OUTPUT_DIR, f"{name}__people__{TODAY}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_sec", "people"])
            w.writerows(zip(np.round(times, 1), counts))
        results.append((name, times, counts))
        print(f"  {name}: 長さ {times[-1] / 60 if len(times) else 0:.1f} 分 / 平均 {counts.mean():.2f} 人 / 最大 {counts.max()} 人")
        for start, mean in top_windows(times, counts):
            print(f"    人が多い区間: START_SECONDS = {int(start)}（{start / 60:.1f} 分）から {WINDOW_SEC} 秒, 平均 {mean:.2f} 人")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 2.4 * len(results)), sharey=True, squeeze=False)
    for ax, (name, times, counts) in zip(axes[:, 0], results):
        ax.plot(times / 60, counts, color="#4C72B0", linewidth=0.6)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("people")
    axes[-1, 0].set_xlabel("minutes from start of file")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"people-timeline__{TODAY}.png")
    fig.savefig(out, dpi=130)
    print(f"完了: {out}")


if __name__ == "__main__":
    main()

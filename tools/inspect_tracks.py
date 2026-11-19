"""
トラックの中身を目で確認するスクリプト

eval_tracking.py が出力した「フレームごとの追跡結果CSV」を読み、
トラックごとの要約表と、各トラックが何を追っていたかが分かる画像を作る。
（例: モデルを変えたら bbox が増えた → 増えたのは見逃していた人か、誤検出か）

使い方:
  1. 下の「設定」の定数を書き換える（引数は不要）
  2. コンテナ内でリポジトリ直下から実行する
       python tools/inspect_tracks.py

出力（tests/<EXPERIMENT_ID>/results/ 以下）:
  - <実験ID>__<設定名>__tracks__<日付>.csv  … トラックごとの要約（長さ・平均スコア・位置・大きさ）
  - <実験ID>__<設定名>__tracks__<日付>.jpg  … トラックごとの切り出し画像の一覧
  - <実験ID>__<設定名>__map__<日付>.jpg     … 動画の1フレームに全トラックの bbox を重ねた地図
"""

import csv
import os
from collections import defaultdict
from datetime import date

import cv2
import numpy as np

# ============================================================
# 設定（ここを書き換えて実行する）
# ============================================================

EXPERIMENT_ID = "lab-video-1106__yolo26-model-size__2026-11-19"
VIDEO_PATH = f"tests/{EXPERIMENT_ID}/input/video.mp4"

# 調べる設定名（eval_tracking.py の CONFIGS の name）
CONFIG_NAMES = ["yolo26n-c010-h060-n060", "yolo26s-c010-h060-n060"]

# eval_tracking.py を実行した日付（フレームごとのCSVのファイル名に入っている）
RUN_DATE = "2026-11-19"

# eval_tracking.py の START_SECONDS と同じ値にする（フレーム番号の起点を合わせるため）
START_SECONDS = 0

# 一覧画像に載せるトラックの最大数（長い順）
MAX_TRACKS_IN_SHEET = 60

# 一覧画像の1マスの大きさ（px）と横に並べる数
TILE_W, TILE_H = 160, 240
TILE_COLS = 10

# 地図の背景に使うフレーム番号
MAP_BACKGROUND_FRAME = 0

# ============================================================
# ここから下は書き換えなくてよい
# ============================================================

TODAY = date.today().isoformat()
RESULT_DIR = f"tests/{EXPERIMENT_ID}/results"


def load_rows(config_name):
    path = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__{config_name}__{RUN_DATE}.csv")
    per_track = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_track[int(r["track_id"])].append({
                "frame": int(r["frame"]), "score": float(r["score"]),
                "box": tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")),
            })
    return per_track


def summarize(per_track, fps, width, height):
    """トラックごとの要約。代表フレームは「映っていた期間の真ん中」のフレーム"""
    table = []
    for tid, rows in per_track.items():
        rows.sort(key=lambda r: r["frame"])
        boxes = np.array([r["box"] for r in rows])
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 / width
        cy = (boxes[:, 1] + boxes[:, 3]) / 2 / height
        span = rows[-1]["frame"] - rows[0]["frame"] + 1
        mid = rows[len(rows) // 2]
        table.append({
            "track_id": tid,
            "first_frame": rows[0]["frame"],
            "last_frame": rows[-1]["frame"],
            "present_sec": round(len(rows) / fps, 2),
            "coverage": round(len(rows) / span, 3),          # 期間中に bbox が出ていた割合（低いほどちらつく）
            "score_mean": round(float(np.mean([r["score"] for r in rows])), 3),
            "center_x": round(float(cx.mean()), 3),           # 画面幅に対する割合
            "center_y": round(float(cy.mean()), 3),
            "move": round(float(np.hypot(cx.max() - cx.min(), cy.max() - cy.min())), 3),  # ほぼ0なら動かない物
            "box_w": round(float((boxes[:, 2] - boxes[:, 0]).mean()), 1),
            "box_h": round(float((boxes[:, 3] - boxes[:, 1]).mean()), 1),
            "_mid_frame": mid["frame"],
            "_mid_box": mid["box"],
        })
    table.sort(key=lambda t: t["present_sec"], reverse=True)
    return table


def read_frames(frame_ids):
    """必要なフレームだけを {番号: 画像} で返す（先頭から順に読むのでシークの誤差がない）"""
    wanted = set(frame_ids)
    last = max(wanted)
    frames = {}
    cap = cv2.VideoCapture(VIDEO_PATH)
    if START_SECONDS:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_SECONDS * fps))
    idx = 0
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames[idx] = img
        idx += 1
    cap.release()
    return frames


def make_sheet(table, frames):
    shown = table[:MAX_TRACKS_IN_SHEET]
    rows_n = (len(shown) + TILE_COLS - 1) // TILE_COLS
    sheet = np.full((max(1, rows_n) * TILE_H, TILE_COLS * TILE_W, 3), 40, np.uint8)
    for i, t in enumerate(shown):
        img = frames[t["_mid_frame"]]
        x0, y0, x1, y1 = (int(v) for v in t["_mid_box"])
        crop = img[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        area_h = TILE_H - 40
        scale = min(TILE_W / crop.shape[1], area_h / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        r, c = divmod(i, TILE_COLS)
        oy, ox = r * TILE_H, c * TILE_W
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        label1 = f"ID{t['track_id']} {t['present_sec']}s"
        label2 = f"sc{t['score_mean']:.2f} cov{t['coverage']:.2f}"
        cv2.putText(sheet, label1, (ox + 4, oy + area_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(sheet, label2, (ox + 4, oy + area_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return sheet


def make_map(table, background):
    img = background.copy()
    rng = np.random.default_rng(0)
    for t in table:
        color = tuple(int(v) for v in rng.integers(64, 256, 3))
        x0, y0, x1, y1 = (int(v) for v in t["_mid_box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, f"{t['track_id']}", (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    for name in CONFIG_NAMES:
        print(f"{name} を集計中...")
        table = summarize(load_rows(name), fps, width, height)
        frames = read_frames([t["_mid_frame"] for t in table[:MAX_TRACKS_IN_SHEET]] + [MAP_BACKGROUND_FRAME])
        base = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__{name}")

        with open(f"{base}__tracks__{TODAY}.csv", "w", newline="", encoding="utf-8") as f:
            fields = [k for k in table[0] if not k.startswith("_")]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(table)
        cv2.imwrite(f"{base}__tracks__{TODAY}.jpg", make_sheet(table, frames))
        cv2.imwrite(f"{base}__map__{TODAY}.jpg", make_map(table, frames[MAP_BACKGROUND_FRAME]))
        print(f"  トラック {len(table)} 本 → {base}__tracks__{TODAY}.csv / .jpg, __map__{TODAY}.jpg")


if __name__ == "__main__":
    main()

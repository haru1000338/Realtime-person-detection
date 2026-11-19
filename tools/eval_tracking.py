"""
ByteTrack しきい値比較スクリプト（人物追跡のちらつき評価）

同じ動画に対して、YOLO の conf と bytetrack.yaml のしきい値を何通りか変えて追跡し、
「bbox のちらつき」「トラックの分断」「ブース境界での行ったり来たり」を数値で比べる。
Re-ID（id_manager.py）と滞在集計（zone_analytics.py）は通さず、トラッカー単体を測る。

使い方:
  1. 下の「設定」の定数を書き換える（引数は不要）
  2. コンテナ内でリポジトリ直下から実行する
       python tools/eval_tracking.py

出力（すべて tests/<EXPERIMENT_ID>/results/ 以下。gitの管理外）:
  - <実験ID>__summary__<日付>.csv          … 設定ごとの指標の一覧（これを比較に使う）
  - <実験ID>__<設定名>__<日付>.csv         … フレームごとの追跡結果（後から別の指標を計算する用）
  - <実験ID>__score-hist__<重み名>__<日付>.csv … 人物検出スコアの分布（重みごと）
  - <実験ID>__score-hist__<日付>.png         … スコア分布の比較グラフ（全重みを重ねて表示）
  - <実験ID>__<設定名>__bytetrack.yaml     … その設定で実際に使った yaml（再現用）

指標はすべて「動画の時間（フレーム数 / fps）」で計算する。
本番の zone_analytics.py は time.time()（実時間）で滞在を測るため、処理速度によって値が変わるが、
このスクリプトは処理速度に影響されない。
"""

import csv
import os
import time
from collections import defaultdict
from datetime import date

import cv2
import numpy as np

# ============================================================
# 設定（ここを書き換えて実行する）
# ============================================================

# 実験ID: <データ>__<検証内容>__<日付>
EXPERIMENT_ID = "lab-video-1106__yolo26-model-size__2026-11-19"

# 入力動画（tests/<実験ID>/input/ に置く想定）
VIDEO_PATH = f"tests/{EXPERIMENT_ID}/input/video.mp4"

# CONFIGS で weights を省略したときに使う YOLO の重み（本番と同じ）
YOLO_WEIGHTS = "yolo26n.pt"

# 動画の何秒目から使うか（0 なら先頭から）。長い録画から人が多い時間帯だけを切り出すときに使う
START_SECONDS = 0

# START_SECONDS から何秒だけ使うか（None なら最後まで）。動作確認のときは 30 などにすると早い
# ※ 出力CSVの frame 列は START_SECONDS を 0 とした番号になる
MAX_SECONDS = None

# 比較する設定。name は出力ファイル名に使う（単語内の区切りは "-"）
#   conf              : model.track() に渡す検出のしきい値（これ未満の検出は ByteTrack に渡らない）
#   track_high_thresh : 1段目の対応づけに使う検出の下限。Lost からの復帰もこれ以上が必要
#   track_low_thresh  : 2段目の対応づけに使う検出の下限
#   new_track_thresh  : 新しいトラックを作る検出の下限
#   weights           : YOLO の重み（省略すると YOLO_WEIGHTS）
CONFIGS = [
    # 基準: 本番と同じ nano + fix/design-gaps の設定（前回の実験と同じ条件。処理時間の比較のため再実行する）
    {"name": "yolo26n-c010-h060-n060", "weights": "yolo26n.pt", "conf": 0.1, "track_high_thresh": 0.6, "track_low_thresh": 0.1, "new_track_thresh": 0.6},
    # モデルを1つ上げる: small + 同じしきい値
    {"name": "yolo26s-c010-h060-n060", "weights": "yolo26s.pt", "conf": 0.1, "track_high_thresh": 0.6, "track_low_thresh": 0.1, "new_track_thresh": 0.6},
    # small はスコアの分布が変わる可能性があるので、しきい値を変えた版も見たいときはコメントを外す
    # {"name": "yolo26s-c010-h050-n050", "weights": "yolo26s.pt", "conf": 0.1, "track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.5},
]

# 全設定で共通の ByteTrack の値（fix/design-gaps の bytetrack.yaml と同じ）
TRACK_BUFFER = 60
MATCH_THRESH = 0.8
FUSE_SCORE = True

# これより短い時間しか映らなかったトラックを「短命トラック」として数える（秒）
SHORT_TRACK_SEC = 1.0

# ブース境界（画面の横幅に対する割合）。本番の zone_analytics.py は 0.5 で左右に分けている
BOOTH_SPLIT_X = 0.5

# ヒステリシスあり集計: 反対側に何秒続けていたら「ブースを移動した」とみなすか
BOOTH_HOLD_SEC = 1.0

# 検出スコア分布を取るときのビン幅
SCORE_BIN = 0.05

# 処理時間の計測から除く最初のフレーム数（GPU の立ち上がりで遅いため）
WARMUP_FRAMES = 30

# ============================================================
# ここから下は書き換えなくてよい
# ============================================================

TODAY = date.today().isoformat()
RESULT_DIR = f"tests/{EXPERIMENT_ID}/results"


def open_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けませんでした: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else None
    return cap, fps, width, max_frames


def iter_frames():
    """(フレーム番号, 画像) を順に返す。START_SECONDS から始め、MAX_SECONDS で打ち切る"""
    cap, fps, _, max_frames = open_video()
    if START_SECONDS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_SECONDS * fps))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and idx >= max_frames):
            break
        yield idx, frame
        idx += 1
    cap.release()


def write_tracker_yaml(cfg):
    path = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__{cfg['name']}__bytetrack.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "tracker_type: bytetrack\n"
            f"track_high_thresh: {cfg['track_high_thresh']}\n"
            f"track_low_thresh: {cfg['track_low_thresh']}\n"
            f"new_track_thresh: {cfg['new_track_thresh']}\n"
            f"track_buffer: {TRACK_BUFFER}\n"
            f"match_thresh: {MATCH_THRESH}\n"
            f"fuse_score: {FUSE_SCORE}\n"
        )
    return path


def weights_of(cfg):
    return cfg.get("weights", YOLO_WEIGHTS)


def run_tracking(cfg):
    """1つの設定で動画を最後まで追跡し、(行のリスト, 総フレーム数, 1フレームの平均処理時間[ms]) を返す"""
    from ultralytics import YOLO

    yaml_path = write_tracker_yaml(cfg)
    # 設定ごとに YOLO を作り直す（トラッカーの状態を前の設定から持ち越さないため）
    model = YOLO(weights_of(cfg))
    rows = []
    n_frames = 0
    elapsed = []
    for idx, frame in iter_frames():
        n_frames = idx + 1
        t0 = time.perf_counter()
        results = model.track(frame, conf=cfg["conf"], classes=[0], persist=True,
                              tracker=yaml_path, verbose=False)
        if idx >= WARMUP_FRAMES:
            elapsed.append(time.perf_counter() - t0)
        boxes = results[0].boxes
        if boxes.id is None:
            continue
        for (x0, y0, x1, y1), tid, score in zip(boxes.xyxy.cpu().numpy(),
                                                boxes.id.cpu().numpy().astype(int),
                                                boxes.conf.cpu().numpy()):
            rows.append({"frame": idx, "track_id": int(tid), "x0": float(x0), "y0": float(y0),
                         "x1": float(x1), "y1": float(y1), "score": float(score)})
        if idx % 300 == 0:
            print(f"  [{cfg['name']}] {idx} フレーム処理済み")
    ms_per_frame = round(float(np.mean(elapsed)) * 1000, 2) if elapsed else 0
    return rows, n_frames, ms_per_frame


def booth_flips(sides, hold_frames):
    """ブース（'A'/'B'）の並びから、移動回数を (生の回数, ヒステリシスありの回数) で返す"""
    raw = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
    held = 0
    current = sides[0] if sides else None
    run_side, run_len = None, 0
    for s in sides[1:]:
        if s == current:
            run_side, run_len = None, 0
            continue
        run_len = run_len + 1 if s == run_side else 1
        run_side = s
        if run_len >= hold_frames:
            held += 1
            current = s
            run_side, run_len = None, 0
    return raw, held


def compute_metrics(rows, n_frames, fps, width):
    """フレームごとの追跡結果から、設定の比較に使う指標を計算する"""
    per_track = defaultdict(list)
    boxes_per_frame = np.zeros(n_frames, dtype=int)
    for r in rows:
        per_track[r["track_id"]].append(r)
        boxes_per_frame[r["frame"]] += 1

    minutes = n_frames / fps / 60 if n_frames else 0
    hold_frames = max(1, int(round(BOOTH_HOLD_SEC * fps)))
    split_px = BOOTH_SPLIT_X * width

    short_tracks = 0
    gap_lengths = []          # 同じIDが消えてから再び出るまでのフレーム数
    raw_flips = held_flips = 0
    for track_rows in per_track.values():
        track_rows.sort(key=lambda r: r["frame"])
        frames = [r["frame"] for r in track_rows]
        if len(frames) / fps < SHORT_TRACK_SEC:
            short_tracks += 1
        gap_lengths += [b - a - 1 for a, b in zip(frames, frames[1:]) if b - a > 1]
        sides = ["A" if (r["x0"] + r["x1"]) / 2 < split_px else "B" for r in track_rows]
        raw, held = booth_flips(sides, hold_frames)
        raw_flips += raw
        held_flips += held

    gaps = np.array(gap_lengths) if gap_lengths else np.array([0])
    return {
        "frames": n_frames,
        "seconds": round(n_frames / fps, 1),
        "tracks_total": len(per_track),
        "tracks_short": short_tracks,
        "gaps_total": len(gap_lengths),
        "gaps_per_min": round(len(gap_lengths) / minutes, 2) if minutes else 0,
        "gap_frames_median": float(np.median(gaps)) if gap_lengths else 0,
        "gap_frames_max": int(gaps.max()) if gap_lengths else 0,
        "boxes_per_frame_mean": round(float(boxes_per_frame.mean()), 3) if n_frames else 0,
        "box_count_change_mean": round(float(np.abs(np.diff(boxes_per_frame)).mean()), 4) if n_frames > 1 else 0,
        "booth_flips_raw": raw_flips,
        "booth_flips_hold": held_flips,
    }


def collect_score_hist(weights):
    """追跡を通さず、YOLO の人物検出スコアの分布を取る（conf=0.1 以上をすべて数える）"""
    from ultralytics import YOLO

    model = YOLO(weights)
    edges = np.arange(0.1, 1.0 + SCORE_BIN, SCORE_BIN)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for idx, frame in iter_frames():
        scores = model.predict(frame, conf=0.1, classes=[0], verbose=False)[0].boxes.conf.cpu().numpy()
        counts += np.histogram(scores, bins=edges)[0]
    return edges, counts


def save_score_hists(hists):
    """hists: {重み名: (edges, counts)}。重みごとの CSV と、全重みを重ねた PNG を保存する"""
    for weights, (edges, counts) in hists.items():
        name = os.path.splitext(os.path.basename(weights))[0]
        path = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__score-hist__{name}__{TODAY}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["score_from", "score_to", "count"])
            for lo, hi, c in zip(edges[:-1], edges[1:], counts):
                w.writerow([f"{lo:.2f}", f"{hi:.2f}", int(c)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for (weights, (edges, counts)), color in zip(hists.items(), colors):
        name = os.path.splitext(os.path.basename(weights))[0]
        ax.stairs(counts, edges, color=color, linewidth=2, label=name)
    for th in (0.6, 0.25):
        ax.axvline(th, color="#C44E52", linestyle="--", linewidth=1)
        ax.text(th, ax.get_ylim()[1] * 0.95, f" {th}", color="#C44E52", va="top")
    ax.legend()
    ax.set_xlabel("person detection score")
    ax.set_ylabel("count (all frames)")
    ax.set_title(EXPERIMENT_ID)
    base = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__score-hist__{TODAY}")
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    _, fps, width, _ = open_video()
    print(f"動画: {VIDEO_PATH}  fps={fps:.2f}  幅={width}px  開始={START_SECONDS}秒  長さ={MAX_SECONDS or '最後まで'}秒")

    hists = {}
    for weights in dict.fromkeys(weights_of(c) for c in CONFIGS):
        print(f"検出スコアの分布を集計中（{weights}）...")
        hists[weights] = collect_score_hist(weights)
    save_score_hists(hists)

    summary = []
    for cfg in CONFIGS:
        print(f"設定 {cfg['name']} で追跡中...")
        rows, n_frames, ms_per_frame = run_tracking(cfg)

        frame_csv = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__{cfg['name']}__{TODAY}.csv")
        with open(frame_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "track_id", "x0", "y0", "x1", "y1", "score"])
            w.writeheader()
            w.writerows(rows)

        metrics = compute_metrics(rows, n_frames, fps, width)
        metrics["ms_per_frame"] = ms_per_frame
        summary.append({"weights": weights_of(cfg), **cfg, **metrics})
        print(f"  → {metrics}")

    summary_csv = os.path.join(RESULT_DIR, f"{EXPERIMENT_ID}__summary__{TODAY}.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"完了: {summary_csv}")


if __name__ == "__main__":
    main()

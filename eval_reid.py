"""
OSNet 特徴量の類似度検証スクリプト

本番パイプライン（filter.py / id_manager.py）を通さず、
FeatureExtractor を直接使って OSNet 単体の性能を測る。

前提:
  - 画像は eval_images/<条件名>/<人物ID>_<任意>.jpg の形式で配置する
  - 人物IDは最初のアンダースコアより前の部分
  - 例) eval_images/same_face_diff_clothes/youtuberA_01.jpg

使い方:
  # ImageNet重み（reid.py と同じ状態）
  python eval_reid.py

  # 特定の重みを指定（model_name はファイル名から自動判定）
  python eval_reid.py --model-path osnet_weights/osnet_x1_0_market1501.pth

  # ディレクトリ内の全重みを順に評価して比較表を出す
  python eval_reid.py --sweep osnet_weights/ --quiet
"""

import argparse
import csv
import itertools
import os
import sys

import cv2
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor
from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ファイル名から model_name を判定するためのパターン（長いものから順に照合）
MODEL_NAME_PATTERNS = [
    "osnet_ain_x1_0", "osnet_ain_x0_75", "osnet_ain_x0_5", "osnet_ain_x0_25",
    "osnet_ibn_x1_0",
    "osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25",
]


def infer_model_name(weight_path):
    """重みファイル名から model_name を推定する"""
    if not weight_path:
        return None
    base = os.path.basename(weight_path)
    for name in MODEL_NAME_PATTERNS:
        if base.startswith(name):
            return name
    return None


def collect_images(root_dir):
    """条件フォルダを走査して画像パスを集める"""
    entries = []
    if not os.path.isdir(root_dir):
        print(f"❌ ディレクトリが見つかりません: {root_dir}")
        sys.exit(1)

    for condition in sorted(os.listdir(root_dir)):
        cond_dir = os.path.join(root_dir, condition)
        if not os.path.isdir(cond_dir):
            continue
        for fname in sorted(os.listdir(cond_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            entries.append({
                "path": os.path.join(cond_dir, fname),
                "condition": condition,
                "person_id": fname.split("_")[0],
                "name": f"{condition}/{fname}",
            })
    return entries


def crop_person(model, img_bgr, conf=0.25):
    """YOLOで人物を検出し、面積最大の領域を切り出す（server.pyのスタッフ登録と同じ方針）"""
    results = model(img_bgr, verbose=False)
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None, 0

    best_crop, max_area, n_person = None, 0, 0
    for box, c, s in zip(boxes.xyxy.cpu().numpy(),
                         boxes.cls.cpu().numpy(),
                         boxes.conf.cpu().numpy()):
        if int(c) != 0 or s < conf:   # class 0 = person
            continue
        n_person += 1
        x0, y0, x1, y1 = map(int, box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img_bgr.shape[1], x1), min(img_bgr.shape[0], y1)
        area = (x1 - x0) * (y1 - y0)
        if area > max_area and (y1 - y0) > 0 and (x1 - x0) > 10:
            max_area = area
            best_crop = img_bgr[y0:y1, x0:x1]

    return best_crop, n_person


def prepare_crops(entries, yolo_model, save_crops_dir=None):
    """YOLOで人物を切り出す。特徴量抽出と分離し、全重みで同じ切り出しを使い回す"""
    prepared = []
    for e in entries:
        img = cv2.imread(e["path"])
        if img is None:
            print(f"⚠️  読み込み失敗のためスキップ: {e['name']}")
            continue

        crop, n_person = crop_person(yolo_model, img)
        if crop is None:
            print(f"⚠️  人物を検出できずスキップ: {e['name']}")
            continue
        if n_person > 1:
            print(f"⚠️  {n_person}人検出（面積最大を採用）: {e['name']}")

        if save_crops_dir:
            os.makedirs(save_crops_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_crops_dir, e["name"].replace("/", "__")), crop)

        e = dict(e)
        e["crop_rgb"] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        prepared.append(e)

    return prepared


def extract_features(prepared, extractor):
    """切り出し済み画像から特徴量を抽出する。id_manager.resolve と同じ正規化を適用"""
    feats = []
    for e in prepared:
        raw = extractor([e["crop_rgb"]])[0]
        e = dict(e)
        e["feature"] = F.normalize(raw.unsqueeze(0), p=2, dim=1).squeeze(0).detach()
        feats.append(e)
    return feats


def cosine(a, b):
    """正規化済みベクトル同士なので内積がそのままコサイン類似度になる"""
    fa, fb = a.float(), b.float()
    if fa.device != fb.device:
        fb = fb.to(fa.device)
    return torch.dot(fa, fb).item()


def build_pairs(feats, threshold):
    """総当たりで全ペアの類似度を計算する"""
    rows = []
    for a, b in itertools.combinations(feats, 2):
        score = cosine(a["feature"], b["feature"])
        cond = (a["condition"] if a["condition"] == b["condition"]
                else f"{a['condition']} × {b['condition']}")
        rows.append({
            "condition": cond,
            "image_a": a["name"],
            "image_b": b["name"],
            "person_a": a["person_id"],
            "person_b": b["person_id"],
            "same_person": a["person_id"] == b["person_id"],
            "score": score,
            "over_threshold": score >= threshold,
        })
    return rows


def summarize(rows, threshold):
    """判定に必要な指標をまとめる"""
    same = [r for r in rows if r["same_person"]]
    diff = [r for r in rows if not r["same_person"]]
    same_min = min((r["score"] for r in same), default=None)
    diff_max = max((r["score"] for r in diff), default=None)
    return {
        "n_same": len(same),
        "n_diff": len(diff),
        "miss": sum(1 for r in same if not r["over_threshold"]),
        "false_match": sum(1 for r in diff if r["over_threshold"]),
        "same_min": same_min,
        "diff_max": diff_max,
        "separated": (same_min is not None and diff_max is not None and same_min > diff_max),
        "overlap": (None if same_min is None or diff_max is None else diff_max - same_min),
    }


def report(rows, threshold, label):
    """1つの重みについての詳細レポートを表示する"""
    same = [r for r in rows if r["same_person"]]
    diff = [r for r in rows if not r["same_person"]]

    def dump(title, subset):
        print("\n" + "-" * 78)
        print(f" {title}  ({len(subset)}ペア)")
        print("-" * 78)
        if not subset:
            print("  （該当なし）")
            return
        for r in sorted(subset, key=lambda x: -x["score"]):
            mark = "★" if r["over_threshold"] else "  "
            print(f" {mark} {r['score']:.4f}  {r['image_a']}  <->  {r['image_b']}")

    dump(f"同一人物ペア（★ = 閾値{threshold}以上 → 正しくマッチ）", same)
    dump(f"別人ペア（★ = 閾値{threshold}以上 → 誤マッチ）", diff)

    print("\n" + "-" * 78)
    print(" 条件別サマリ")
    print("-" * 78)
    print(f"{'条件':<40} {'種別':<6} {'件数':>4} {'最小':>7} {'中央':>7} {'最大':>7}")
    for cond in sorted({r["condition"] for r in rows}):
        for name, flag in (("同一", True), ("別人", False)):
            sub = sorted(r["score"] for r in rows
                         if r["condition"] == cond and r["same_person"] == flag)
            if not sub:
                continue
            disp = cond if len(cond) <= 38 else cond[:37] + "…"
            print(f"{disp:<40} {name:<6} {len(sub):>4} "
                  f"{sub[0]:>7.4f} {sub[len(sub) // 2]:>7.4f} {sub[-1]:>7.4f}")


def print_judgement(s, threshold):
    """閾値に対する判定を表示する"""
    print("\n" + "-" * 78)
    print(f" 閾値 {threshold} での判定")
    print("-" * 78)
    print(f" 見失い   : {s['miss']} / {s['n_same']} 件（同一人物なのに閾値未満）")
    print(f" 誤マッチ : {s['false_match']} / {s['n_diff']} 件（別人なのに閾値以上）")
    if s["same_min"] is None or s["diff_max"] is None:
        return
    print(f" 同一人物ペアの最小 : {s['same_min']:.4f}")
    print(f" 別人ペアの最大     : {s['diff_max']:.4f}")
    if s["separated"]:
        print(f"\n ✅ 分離できています。"
              f"閾値を {s['diff_max']:.2f}〜{s['same_min']:.2f} に置けば全て正解します。")
    else:
        print(f"\n ⚠️  分布が重なっています（重なり幅 {s['overlap']:.4f}）。"
              f"閾値をどこに置いても誤りが残ります。")


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"💾 {path}")


def build_targets(args):
    """評価する (ラベル, model_name, 重みパス) のリストを作る"""
    if not args.sweep:
        name = args.model_name or infer_model_name(args.model_path) or "osnet_x1_0"
        label = (os.path.splitext(os.path.basename(args.model_path))[0]
                 if args.model_path else "imagenet_baseline")
        return [(label, name, args.model_path)]

    if not os.path.isdir(args.sweep):
        print(f"❌ ディレクトリが見つかりません: {args.sweep}")
        sys.exit(1)

    # ImageNet重み（現行の reid.py と同じ状態）をベースラインとして先頭に置く
    targets = [("imagenet_baseline", "osnet_x1_0", "")]
    for fname in sorted(os.listdir(args.sweep)):
        if not fname.endswith(".pth"):
            continue
        name = infer_model_name(fname)
        if name is None:
            print(f"⚠️  model_name を判定できずスキップ: {fname}")
            continue
        targets.append((os.path.splitext(fname)[0], name, os.path.join(args.sweep, fname)))
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_images", help="画像のルートディレクトリ")
    ap.add_argument("--yolo", default="yolo26n.pt", help="YOLOモデルのパス")
    ap.add_argument("--model-name", default=None,
                    help="OSNetのアーキテクチャ名。省略時は重みファイル名から自動判定")
    ap.add_argument("--model-path", default="",
                    help="Re-ID学習済み重みのパス。省略時はImageNet重み")
    ap.add_argument("--sweep", default=None,
                    help="指定ディレクトリ内の全 .pth を順に評価して比較表を出す")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="本番の similarity_threshold と同じ値")
    ap.add_argument("--csv-dir", default="eval_results", help="結果CSVの出力先ディレクトリ")
    ap.add_argument("--save-crops", default=None, help="切り出し画像の保存先")
    ap.add_argument("--quiet", action="store_true", help="ペアの明細を表示しない")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 画像の準備（切り出しは1回だけ行い、全重みで使い回す）----
    entries = collect_images(args.dir)
    if len(entries) < 2:
        print(f"❌ 画像が2枚未満です（{len(entries)}枚）。配置を確認してください。")
        sys.exit(1)

    print("=" * 78)
    print(" OSNet 特徴量 類似度検証")
    print("=" * 78)
    conditions = sorted({e["condition"] for e in entries})
    persons = sorted({e["person_id"] for e in entries})
    print(f"\n📁 画像 {len(entries)}枚 / 条件 {len(conditions)}種 / 人物ID {len(persons)}種")
    for c in conditions:
        sub = [e for e in entries if e["condition"] == c]
        ids = sorted({e["person_id"] for e in sub})
        print(f"   - {c}: {len(sub)}枚  人物ID={', '.join(ids)}")

    print("\n🔍 YOLOで人物を切り出します...")
    prepared = prepare_crops(entries, YOLO(args.yolo), args.save_crops)
    print(f"✅ {len(prepared)}枚を切り出しました（device={device}）")

    # ---- 各重みで評価 ----
    results = []
    for label, model_name, model_path in build_targets(args):
        print("\n" + "#" * 78)
        print(f"# {label}")
        print(f"#   model_name = {model_name}")
        print(f"#   model_path = {model_path or '(未指定 → ImageNet重み)'}")
        print("#" * 78)

        try:
            extractor = FeatureExtractor(
                model_name=model_name, model_path=model_path, device=device
            )
            feats = extract_features(prepared, extractor)
        except Exception as exc:
            print(f"❌ 評価に失敗しました: {exc}")
            continue
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

        rows = build_pairs(feats, args.threshold)
        s = summarize(rows, args.threshold)

        if not args.quiet:
            report(rows, args.threshold, label)
        print_judgement(s, args.threshold)
        save_csv(rows, os.path.join(args.csv_dir, f"{label}.csv"))
        results.append((label, s))

    # ---- 比較表 ----
    if len(results) > 1:
        print("\n" + "=" * 96)
        print(" 重みごとの比較")
        print("=" * 96)
        print(f"{'重み':<34} {'見失い':>8} {'誤マッチ':>10} "
              f"{'同一min':>9} {'別人max':>9} {'重なり':>9}  判定")
        print("-" * 96)
        for label, s in results:
            disp = label if len(label) <= 32 else label[:31] + "…"
            same_min = f"{s['same_min']:.4f}" if s["same_min"] is not None else "-"
            diff_max = f"{s['diff_max']:.4f}" if s["diff_max"] is not None else "-"
            overlap = f"{s['overlap']:+.4f}" if s["overlap"] is not None else "-"
            mark = "✅ 分離" if s["separated"] else "⚠️ 重なり"
            print(f"{disp:<34} {s['miss']:>4}/{s['n_same']:<3} "
                  f"{s['false_match']:>5}/{s['n_diff']:<4} "
                  f"{same_min:>9} {diff_max:>9} {overlap:>9}  {mark}")
        print("-" * 96)
        print(" ※ 重なり = 別人max - 同一min。負の値なら分離できている（小さいほど良い）")


if __name__ == "__main__":
    main()
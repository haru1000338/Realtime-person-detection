#!/usr/bin/env python3
"""Market-1501 から評価用の画像を選び、実験ディレクトリにコピーする。

下の「設定」を書き換えて、`python tools/select_images.py` と実行する。
SEED を固定しているので、同じ設定なら何度実行しても同じ画像が選ばれる。
"""

import csv
import os
import random
import re
import shutil
from collections import defaultdict

# =========================== 設定（ここだけ書き換える）===========================

SRC = "tests/_datasets/Market-1501-v15.09.15/bounding_box_test"
DST = "tests/market1501__unseen-data-check-100__2026-10-01/input/market1501_bounding_box_test"

NUM_PERSONS = 20      # 選ぶ人数
PER_PERSON = 5        # 1人あたりの枚数
MIN_CAMERAS = 2       # 1人あたり最低これだけのカメラにまたがらせる
SEED = 20261001       # 乱数シード。変えると選ばれる画像が変わる

# ==============================================================================

# 0001_c1s1_000151_01.jpg → 人物ID 0001 / カメラ 1 / シーケンス 1
PATTERN = re.compile(r"^(\d{4})_c(\d+)s(\d+)_")


def collect():
    """データセットを走査し、人物ID -> 画像ファイル名のリストを作る。

    除外するもの:
      -1 で始まる  … 検出器の誤検出（distractor）
      0000 で始まる … 評価対象外のジャンク画像
      .jpg 以外     … Thumbs.db など
    """
    persons = defaultdict(list)
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".jpg"):
            continue
        m = PATTERN.match(name)
        if not m or m.group(1) == "0000":
            continue          # 先頭が -1 の場合は PATTERN に一致しないので除外される
        persons[m.group(1)].append(name)
    return persons


def cameras_of(names):
    """画像名のリストから、使われているカメラ番号の集合を返す。"""
    return {PATTERN.match(n).group(2) for n in names}


def pick_for_person(names, rng):
    """1人分の画像を PER_PERSON 枚選ぶ。カメラが MIN_CAMERAS 未満なら選び直す。

    基本はただのランダム抽出。ただし同じカメラの連続フレームばかりだと
    「ほぼ同じ画像」を比べることになるため、カメラ数の条件だけ満たすまで引き直す。
    """
    for _ in range(100):
        picked = rng.sample(names, PER_PERSON)
        if len(cameras_of(picked)) >= MIN_CAMERAS:
            return picked
    return None               # 100回引いても条件を満たさなければ、この人物は諦める


def main():
    if os.path.exists(DST) and os.listdir(DST):
        raise SystemExit(f"コピー先に既にファイルがあります: {DST}\n"
                         f"設定の DST を確認するか、不要なら削除してください。")

    rng = random.Random(SEED)
    persons = collect()

    # 条件を満たす人物だけを候補にする
    eligible = sorted(
        pid for pid, names in persons.items()
        if len(names) >= PER_PERSON and len(cameras_of(names)) >= MIN_CAMERAS
    )
    print(f"候補人物: {len(eligible)}人（全{len(persons)}人中）")
    if len(eligible) < NUM_PERSONS:
        raise SystemExit(f"候補が {NUM_PERSONS} 人に足りません")

    selected = {}
    for pid in rng.sample(eligible, NUM_PERSONS):
        picked = pick_for_person(persons[pid], rng)
        if picked:
            selected[pid] = sorted(picked)
    if len(selected) < NUM_PERSONS:
        raise SystemExit(f"条件を満たす選定ができたのは {len(selected)} 人だけでした")

    # コピーと、選定結果の記録
    os.makedirs(DST, exist_ok=True)
    rows = []
    for pid in sorted(selected):
        for name in selected[pid]:
            shutil.copy2(os.path.join(SRC, name), os.path.join(DST, name))
            m = PATTERN.match(name)
            rows.append([name, pid, m.group(2), m.group(3)])

    record = os.path.join(os.path.dirname(DST), "selected_images.csv")
    with open(record, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "person_id", "camera", "sequence"])
        w.writerows(rows)

    # 確認用の要約
    total = len(rows)
    cam_dist = defaultdict(int)
    for pid in selected:
        cam_dist[len(cameras_of(selected[pid]))] += 1
    print(f"選定  : {len(selected)}人 × {PER_PERSON}枚 = {total}枚")
    print(f"ペア数: {total * (total - 1) // 2}組"
          f"（同一人物 {len(selected) * PER_PERSON * (PER_PERSON - 1) // 2}）")
    for k in sorted(cam_dist):
        print(f"  カメラ{k}台にまたがる人: {cam_dist[k]}人")
    print(f"コピー先: {DST}")
    print(f"選定記録: {record}")


if __name__ == "__main__":
    main()
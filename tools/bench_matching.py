"""
Re-ID 照合（_find_best_match）の速度比較スクリプト

保存している人数を変えながら、
  - 旧実装: 1件ずつ reid.compare_features() で比べる（.item() のたびに GPU と同期）
  - 新実装: id_manager.py の IDManager._find_best_match()（テンソル化版）
の処理時間を測り、両者が同じ人物IDを選ぶかも確認する。

特徴量はランダムな512次元ベクトルで作る（OSNet は通さない）。
本番に近い分布にするため、1人ぶんの特徴量は「その人の中心ベクトル + 小さなノイズ」で作る。

使い方:
  コンテナ内でリポジトリ直下から実行する（引数は不要。設定は下の定数）
    python tools/bench_matching.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reid  # noqa: E402  OSNet の読み込みが走るが、使うのは compare_features だけ
from id_manager import IDManager  # noqa: E402

# ============================================================
# 設定（ここを書き換えて実行する）
# ============================================================

# 保存している人数（この人数ごとに測る）
N_PEOPLE_LIST = [10, 50, 100, 200, 500]

# 1人あたりの特徴量の数（本番の MAX_POOL_SIZE）
FEATS_PER_PERSON = 5

# 1つの人数につき何回照合して平均をとるか
REPEAT = 50

# 特徴量の次元（OSNet は 512）
DIM = 512

# ============================================================


def old_find_best_match(query_feat, pool_dict):
    """以前の実装（比較用にそのまま残したもの）"""
    best_id = None
    best_score = 0.0
    for person_id, feature_list in pool_dict.items():
        for stored_feat in feature_list:
            score = reid.compare_features(query_feat, stored_feat)
            if score > best_score:
                best_score = score
                best_id = person_id
    return best_id, best_score


def make_pool(n_people, device, gen):
    pool = {}
    for i in range(n_people):
        center = torch.randn(DIM, generator=gen)
        pool[f"R{i:03d}"] = [(center + 0.3 * torch.randn(DIM, generator=gen)).to(device)
                             for _ in range(FEATS_PER_PERSON)]
    return pool


def timed(fn, *args):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    new_find_best_match = IDManager._find_best_match  # self は使っていないので None を渡す
    print(f"device={device}  1人あたり {FEATS_PER_PERSON} 個  各 {REPEAT} 回の平均\n")
    print(f"{'人数':>6} {'特徴量数':>8} {'旧[ms]':>9} {'新[ms]':>9} {'倍率':>7} {'ID一致':>8} {'スコア差の最大':>14}")

    for n in N_PEOPLE_LIST:
        pool = make_pool(n, device, gen)
        ids = list(pool)
        old_ms, new_ms, same, max_diff = [], [], 0, 0.0
        for r in range(REPEAT + 3):
            # 保存済みの誰かに似たクエリ（同一人物の別の姿を想定）
            target = pool[ids[r % n]][0]
            query = (target + 0.3 * torch.randn(DIM, generator=gen).to(device))
            (oid, osc), ot = timed(old_find_best_match, query, pool)
            (nid, nsc), nt = timed(new_find_best_match, None, query, pool)
            if r < 3:  # 最初の数回は GPU の立ち上がりで遅いので除く
                continue
            old_ms.append(ot)
            new_ms.append(nt)
            same += int(oid == nid)
            max_diff = max(max_diff, abs(osc - nsc))
        o, w = sum(old_ms) / len(old_ms), sum(new_ms) / len(new_ms)
        print(f"{n:>6} {n * FEATS_PER_PERSON:>8} {o:>9.2f} {w:>9.2f} {o / w:>6.1f}x {same:>4}/{REPEAT} {max_diff:>14.2e}")


if __name__ == "__main__":
    main()

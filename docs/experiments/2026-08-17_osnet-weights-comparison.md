# OSNet 類似度検証（重み7種の比較 / 入力形式の影響）
 
実施日: 2026-08-17 / ブランチ: `feature/reid-evaluation`
 
前回（`2026-08-07_osnet-imagenet-baseline.md`）で ImageNet 重みが使われていることが判明したため、
Re-ID 学習済み重みに差し替えて比較した。あわせて、切り出し範囲（バストアップ／全身）の影響も測定した。
 
## 条件
 
| 項目 | 値 |
|------|-----|
| 特徴量 | 512次元 / 入力 256×128 |
| 切り出し | `yolo26n.pt` で検出、面積最大の1人、conf=0.25 |
| 正規化 | L2正規化（`F.normalize`） |
| 類似度 | コサイン類似度 |
| 閾値 | 0.70（`id_manager` の現行値） |
| 実行 | `python eval_reid.py --sweep osnet_weights/ --save-crops crops/` |
 
切り出しは1回だけ実行し、全重みで同一の切り出し画像を使用した。
 
## 使用画像
 
12枚 / 9人物 / 66ペア（同一人物 4件、別人 62件）
 
| フォルダ | 枚数 | 人物 | 内容 |
|----------|-----:|-----:|------|
| `diff_face_same_clothes/` | 4 | 4 | バレー代表ユニフォーム。**バストアップ**。白背景 |
| `diff_face_same_clothes_full/` | 3 | 3 | サッカー代表ユニフォーム。**全身**。グレー背景 |
| `same_all_diff_angle/` | 3 | 1 | 紺スーツ。腕組み／顎に手／前傾（姿勢違い） |
| `same_face_diff_clothes/` | 2 | 1 | 水色シャツ+ジーンズ ／ 黒ジャケット+黒パンツ。白背景 |
 
※ 画像は `.gitignore` により追跡外
 
## 検証した重み
 
| # | ラベル | model_name | 学習データ | CSV |
|---|--------|------------|------------|-----|
| 1 | `imagenet_baseline` | osnet_x1_0 | ImageNet（Re-ID学習なし） | `results/imagenet_baseline.csv` |
| 2 | `osnet_ain_x1_0_msmt17_ca` | osnet_ain_x1_0 | MSMT17 (combineall) | `results/osnet_ain_x1_0_msmt17_ca.csv` |
| 3 | `osnet_x0_5_msmt17_ca` | osnet_x0_5 | MSMT17 (combineall) | `results/osnet_x0_5_msmt17_ca.csv` |
| 4 | `osnet_x0_75_msmt17_ca` | osnet_x0_75 | MSMT17 (combineall) | `results/osnet_x0_75_msmt17_ca.csv` |
| 5 | `osnet_x1_0_market1501` | osnet_x1_0 | Market-1501 | `results/osnet_x1_0_market1501.csv` |
| 6 | `osnet_x1_0_msmt17` | osnet_x1_0 | MSMT17 | `results/osnet_x1_0_msmt17.csv` |
| 7 | `osnet_x1_0_msmt17_ca` | osnet_x1_0 | MSMT17 (combineall) | `results/osnet_x1_0_msmt17_ca.csv` |
| — | `osnet_ain_x1_0_msdc` | osnet_ain_x1_0 | MS+D+C → M | **読み込み失敗**（記録事項参照） |
 
### CSVのカラム
 
| カラム | 内容 |
|--------|------|
| `condition` | 条件フォルダ名。異なるフォルダ間の比較は `A × B` 形式 |
| `image_a` / `image_b` | 比較した2枚のパス |
| `person_a` / `person_b` | 人物ID（ファイル名の最初のアンダースコアより前） |
| `same_person` | 同一人物か（True / False） |
| `score` | コサイン類似度 |
| `over_threshold` | 閾値0.70以上か（True / False） |
 
## 結果
 
### 主要4条件のスコア（中央値）
 
| 重み | 同ユニ・バスト<br>（別人） | 同ユニ・全身<br>（別人） | 姿勢違い<br>（同一） | 服装違い<br>（同一） |
|------|----:|----:|----:|----:|
| imagenet_baseline | 0.9319 | 0.8034 | 0.8338 | 0.7692 |
| osnet_ain_x1_0_msmt17_ca | 0.9594 | 0.8779 | 0.7258 | 0.5807 |
| osnet_x0_5_msmt17_ca | 0.9534 | 0.8095 | 0.7702 | 0.6608 |
| osnet_x0_75_msmt17_ca | 0.9446 | 0.8659 | 0.8261 | 0.4777 |
| osnet_x1_0_market1501 | 0.9612 | 0.8878 | 0.8941 | 0.5460 |
| osnet_x1_0_msmt17 | 0.9602 | 0.8461 | 0.7569 | 0.5607 |
| osnet_x1_0_msmt17_ca | 0.9390 | 0.8471 | 0.7531 | 0.4697 |
 
### 同一ユニフォーム・別人（バストアップ vs 全身）
 
| 重み | バスト最小 | バスト中央 | バスト最大 | 全身最小 | 全身中央 | 全身最大 | 中央値の差 |
|------|----:|----:|----:|----:|----:|----:|----:|
| imagenet_baseline | 0.9020 | 0.9319 | 0.9363 | 0.7806 | 0.8034 | 0.8098 | -0.1285 |
| osnet_ain_x1_0_msmt17_ca | 0.9408 | 0.9594 | 0.9671 | 0.8427 | 0.8779 | 0.8782 | -0.0815 |
| osnet_x0_5_msmt17_ca | 0.9363 | 0.9534 | 0.9671 | 0.7989 | 0.8095 | 0.8359 | -0.1439 |
| osnet_x0_75_msmt17_ca | 0.9245 | 0.9446 | 0.9478 | 0.8555 | 0.8659 | 0.8789 | -0.0787 |
| osnet_x1_0_market1501 | 0.9510 | 0.9612 | 0.9762 | 0.8555 | 0.8878 | 0.9133 | -0.0734 |
| osnet_x1_0_msmt17 | 0.9326 | 0.9602 | 0.9645 | 0.8356 | 0.8461 | 0.8752 | -0.1141 |
| osnet_x1_0_msmt17_ca | 0.9066 | 0.9390 | 0.9599 | 0.8166 | 0.8471 | 0.8644 | -0.0919 |
 
### 閾値 0.70 での判定
 
| 重み | 見失い | 誤マッチ | 同一min | 別人max | 重なり幅 |
|------|------:|--------:|--------:|--------:|--------:|
| imagenet_baseline | 0 / 4 | 10 / 62 | 0.7586 | 0.9363 | +0.1777 |
| osnet_ain_x1_0_msmt17_ca | 1 / 4 | 9 / 62 | 0.5807 | 0.9671 | +0.3863 |
| osnet_x0_5_msmt17_ca | 1 / 4 | 9 / 62 | 0.6608 | 0.9671 | +0.3063 |
| osnet_x0_75_msmt17_ca | 1 / 4 | 9 / 62 | 0.4777 | 0.9478 | +0.4701 |
| osnet_x1_0_market1501 | 1 / 4 | 10 / 62 | 0.5460 | 0.9762 | +0.4302 |
| osnet_x1_0_msmt17 | 1 / 4 | 9 / 62 | 0.5607 | 0.9645 | +0.4037 |
| osnet_x1_0_msmt17_ca | 1 / 4 | 9 / 62 | 0.4697 | 0.9599 | +0.4902 |
 
全7重みで分布が重なった（分離できた重みはなし）。
 
## 記録事項
 
- **全身画像3枚には JFA の「OFFICIAL GOODS」ロゴが写り込んでいた。** 3枚とも同一デザイン・同一位置のため、`diff_face_same_clothes_full` のスコアが上振れしている可能性がある
- `osnet_ain_x1_0_msdc.pth` は読み込みに失敗した。PyTorch 2.6 で `torch.load` の `weights_only` 既定値が `True` に変更されたため、numpy スカラーを含むチェックポイントが弾かれる（`GLOBAL numpy.core.multiarray.scalar was not an allowed global`）
- `jpn20_01.jpg` で「2人検出」の警告（面積最大を採用）。前回から継続
- 全身画像1枚は `.webp` 形式。読み込みは成功した
- 同一人物ペアが4件のみで、そのうち3件が同一人物（modelB）の姿勢違い。統計的には不足
## 考察
 
（追記予定）
 
## 関連ファイル
 
| ファイル | 内容 |
|----------|------|
| `eval_reid.py` | 検証スクリプト（`--sweep` 対応版） |
| `results/*.csv` | 重みごとの全66ペアの生データ（7ファイル） |
| `osnet_weights/` | 検証に使用した重み（`.gitignore` により追跡外） |
| `eval_images/` | 使用画像（同上） |
| `crops/` | YOLO による切り出し結果（同上） |
 

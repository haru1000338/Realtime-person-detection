# 手動切り出し画像の追加による比較

- 実施日: 2026-09-10（`eval_results/` と `run_v2.log` の更新日時より。実行日そのものは未確認）
- ブランチ: `feature/reid-evaluation`
- コミット: 8364d66

## 目的

入力画像に人以外の領域（余白）が多く、精度に影響しているのではないか、という
メンターからの指摘を確認する。

## 方法

既存の12枚に対し、手作業で余白を取り除いた画像7枚（`*-crop.jpg`）を追加し、
同じ7種類の重みで再評価した。

## 実施後に判明したこと

**`eval_reid.py` は入力画像をYOLOで検出・切り出ししてからOSNetに渡す作りになっていた。**
実施時点ではこれを認識していなかった。

そのため本実験は「余白の有無が精度に与える影響」ではなく、
**「手作業で余白を削った場合と、YOLOの切り出しに任せた場合で、OSNetへの入力が変わるか」**
を測定したものと解釈する。

この問いには、元画像と手動切り出し版のペア（7組）の類似度で答えられる。
**本記録の時点では未集計。**

## 条件

| 項目 | 値 |
|---|---|
| モデル | 重みごとに `eval_reid.py` が自動判定 |
| 重み | 7種（ImageNet + Re-ID用6種） |
| 判定閾値 | 0.70 |
| 検出器 | `yolo26n.pt` |
| デバイス | cuda |

`osnet_ain_x1_0_msdc.pth` は読み込みに失敗した（`run_v2.log` に記録あり）。

## 使用画像

19枚 / 9人物 / 171ペア（同一人物 11件、別人 160件）

| 条件フォルダ | 枚数 | 内容 |
|---|---|---|
| `diff_face_same_clothes` | 8 | 同一ユニフォームの別人（バストアップ）。うち4枚は手動切り出し版 |
| `diff_face_same_clothes_full` | 6 | 同一ユニフォームの別人（全身）。うち3枚は手動切り出し版 |
| `same_all_diff_angle` | 3 | 同一人物・姿勢違い |
| `same_face_diff_clothes` | 2 | 同一人物・服装違い |

`web-photos__weights-comparison__2026-08-17` の12枚に、以下7枚を追加したもの。

```
diff_face_same_clothes/jpn11_01-crop.jpg
diff_face_same_clothes/jpn16_01-crop.jpg
diff_face_same_clothes/jpn20_01-crop.jpg
diff_face_same_clothes/jpn21_01-crop.jpg
diff_face_same_clothes_full/socA_01-crop.jpg
diff_face_same_clothes_full/socB_01-crop.jpg
diff_face_same_clothes_full/socC_01-crop.jpg
```

人物IDは最初のアンダースコアより前の部分のため、元画像と手動切り出し版は
**同一人物ペア**として扱われている。

## 結果

**未集計。** 生データは関連ファイルのCSVに保存されている。

## 記録事項

- 実行コマンドは記録されていない。`run_v2.log` には標準出力のみが保存されており、
  引数は残っていない（出力先から `--sweep osnet_weights/ --save-crops crops_v2/` と推定）
- 本実験の実行により、`eval_results/` にあった
  `web-photos__weights-comparison__2026-08-17` の生データは上書きされた。
  同実験の結果が残っているのは `docs/experiments/results/` にコピーしていたため
- CSVの `condition` 列は、異なる条件同士のペアでは `A × B` という複合名になる。
  そのため `condition` の一意な値は10種（入力の条件フォルダは4種）

## 考察

### 確認できたこと

- 19枚・171ペアの評価が完了し、7種の重みそれぞれについて生データが得られた

### 未検証・推測

- 「手動切り出しとYOLO切り出しで入力が変わるか」は未集計
- メンターの指摘した「余白」が、YOLOの枠の外側を指すのか、
  256×128へのリサイズ時の変形を指すのかは未確認。後者であれば本実験では測れていない

## 関連ファイル

- 結果CSV: `results/web-photos__manual-crop-comparison__*__2026-09-10.csv`（7ファイル）
- 実行ログ: `run_v2.log`（gitでは追跡されていない）
- 入力画像・切り出し画像: gitでは追跡されていない

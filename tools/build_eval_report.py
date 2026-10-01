#!/usr/bin/env python3
"""eval_reid.py が出力した CSV から、評価レポート Excel を作る。

下の「設定」を書き換えて、`python tools/build_eval_report.py` と実行する。
eval_reid.py には触らないので、このスクリプトを何度変えても score は変わらない。
"""

import csv
import math
import os
import statistics

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# =========================== 設定（ここだけ書き換える）===========================

CSV = "docs/experiments/results/market1501__unseen-data-check-100__osnet_x1_0_msmt17__2026-10-01.csv"
OUT = "tests/market1501__unseen-data-check-100__2026-10-01/results/report.xlsx"
THRESHOLD = 0.70
BUILD_MATRIX = True   # 類似度行列シート。画像が多いと巨大になるので100枚では False 推奨

# Sheet2 に書く実験条件。ラベルと値をそのまま並べる
META = [
    ("実験名", "Market-1501 test split によるベースライン検証（100枚）"),
    ("実施日", "2026-10-01"),
    ("目的", "要記入"),
    ("仮説", "要記入"),
    ("モデル", "osnet_x1_0"),
    ("重みファイル", "osnet_x1_0_msmt17.pth"),
    ("重みの学習データ", "MSMT17（126,441枚 / 4,101人）"),
    ("検出器", "yolo26n.pt"),
    ("デバイス", "cuda"),
    ("ブランチ", "feature/reid-eval-market1501-100"),
    ("コミット", "要記入（git rev-parse --short HEAD）"),
    ("データセット", "Market-1501 v15.09.15 bounding_box_test"),
    ("選定方法", "要記入"),
    ("構成", "要記入"),
    ("リークの有無", "なし（MSMT17で学習、Market-1501で評価）"),
]

# ==============================================================================

BIN = 0.05
FONT = "Yu Gothic"
HEADERS = ["condition", "image_a", "image_b", "person_a", "person_b",
           "same_person", "score", "over_threshold"]


def read_rows(path):
    """CSV を読み、score の降順に並べて返す。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [{
            "condition": r["condition"],
            "image_a": r["image_a"],
            "image_b": r["image_b"],
            # person列をそのまま使うと Excel で '0728' が 728 になるため文字列で持つ
            "person_a": str(r["person_a"]),
            "person_b": str(r["person_b"]),
            "same_person": r["same_person"].strip().lower() == "true",
            "score": float(r["score"]),
            "over_threshold": r["over_threshold"].strip().lower() == "true",
        } for r in csv.DictReader(f)]
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def label(path):
    """'.../0728_c1s4_002381_01.jpg' -> '0728_c1s4_002381'（行列の見出し用）"""
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    return "_".join(parts[:3]) if len(parts) >= 4 else "_".join(parts)


def sheet1(wb, rows):
    """生データ・ビン集計・統計・混同行列・グラフ。"""
    ws = wb.create_sheet("Sheet1")
    last = len(rows) + 1
    F, G = f"$F$2:$F${last}", f"$G$2:$G${last}"   # same_person列 と score列

    ws.append(HEADERS)
    for r in rows:
        ws.append([r[h] for h in HEADERS])
    for row in ws.iter_rows(min_row=2, max_row=last, min_col=4, max_col=5):
        for c in row:
            c.number_format = "@"                 # 人物IDの先頭ゼロを保持する
    for c in ws[1]:
        c.font = Font(name=FONT, bold=True)

    # 誤判定した行（same_person と over_threshold が食い違う行）を黄色で塗る
    ws.conditional_formatting.add(
        f"A2:H{last}",
        FormulaRule(formula=["$F2 <> ($G2 >= $Z$3)"],
                    fill=PatternFill("solid", start_color="FFFFFF00", end_color="FFFFFF00"),
                    stopIfTrue=True),
    )

    # 全体の中央に位置する行を青で塗る。
    # score の降順に並んでいるので、ここが全ペアの中心（中央値）にあたる。
    # 偶数件のときは中央が2行になるため、その両方を塗る。
    total = len(rows)
    middle = ([(total + 1) // 2 + 1] if total % 2
              else [total // 2 + 1, total // 2 + 2])
    blue = PatternFill("solid", start_color="FFBDD7EE", end_color="FFBDD7EE")
    for row in middle:
        for col in "ABCDEFGH":
            ws[f"{col}{row}"].fill = blue

    ws["J1"] = "黄色 = 閾値(Z3)で誤判定した行"
    ws["J2"] = "青 = 全ペアの中央に位置する行"

    # --- ビンごとの相対度数（件数ではなく割合。各群の合計が 1 になる）---
    lo = math.floor(min(r["score"] for r in rows) / BIN) * BIN
    hi = math.ceil(max(r["score"] for r in rows) / BIN) * BIN
    bins = [round(lo + i * BIN, 2) for i in range(int(round((hi - lo) / BIN)))]

    ws["K1"], ws["L1"], ws["M1"], ws["N1"] = "下限", "上限", "同一人物", "別人"
    for i, lower in enumerate(bins):
        n = i + 2
        ws[f"K{n}"] = lower
        ws[f"L{n}"] = f"=K{n}+{BIN}"
        for col, flag in (("M", "TRUE"), ("N", "FALSE")):
            ws[f"{col}{n}"] = (f'=COUNTIFS({F},{flag},{G},">="&$K{n},{G},"<"&$L{n})'
                               f'/COUNTIF({F},{flag})')
            ws[f"{col}{n}"].number_format = "0.000"

    end = len(bins) + 2
    ws[f"L{end}"] = "合計"
    ws[f"M{end}"] = f"=SUM(M2:M{end - 1})"
    ws[f"N{end}"] = f"=SUM(N2:N{end - 1})"
    ws[f"P{end}"] = "※ 両方 1.000 ならビンが全データを覆えている"

    # --- 群ごとの統計 ---
    # 中央値だけは条件付き MEDIAN が配列数式になり壊れるため Python で計算する
    for j, name in enumerate(["区分", "件数", "最小", "中央値", "平均", "最大"]):
        ws.cell(row=1, column=16 + j, value=name).font = Font(name=FONT, bold=True)
    for i, (name, flag) in enumerate([("同一人物", "TRUE"), ("別人", "FALSE")]):
        n = i + 2
        values = [r["score"] for r in rows if r["same_person"] == (flag == "TRUE")]
        ws[f"P{n}"] = name
        ws[f"Q{n}"] = f"=COUNTIF({F},{flag})"
        ws[f"R{n}"] = f"=_xlfn.MINIFS({G},{F},{flag})"
        ws[f"S{n}"] = statistics.median(values) if values else None
        ws[f"T{n}"] = f"=AVERAGEIF({F},{flag},{G})"
        ws[f"U{n}"] = f"=_xlfn.MAXIFS({G},{F},{flag})"
        for col in "RSTU":
            ws[f"{col}{n}"].number_format = "0.0000"
    ws["P4"] = "※ 中央値はスクリプトで算出（条件付き MEDIAN は数式にできないため）"

    # --- 混同行列（閾値セル Z3 を変えれば Excel 上で再計算される）---
    ws["Y3"], ws["Z3"] = "閾値", THRESHOLD
    ws["AA7"], ws["AB7"] = "予測:同一人物", "予測:別人"
    ws["Z8"], ws["Z9"] = "実際:同一人物", "実際:別人"
    ws["AA8"] = f'=COUNTIFS({F},TRUE,{G},">="&$Z$3)'     # TP
    ws["AB8"] = f'=COUNTIFS({F},TRUE,{G},"<"&$Z$3)'      # FN
    ws["AA9"] = f'=COUNTIFS({F},FALSE,{G},">="&$Z$3)'    # FP
    ws["AB9"] = f'=COUNTIFS({F},FALSE,{G},"<"&$Z$3)'     # TN
    for n, name, f in ((11, "適合率", "AA8/(AA8+AA9)"),
                       (12, "再現率", "AA8/(AA8+AB8)"),
                       (13, "F1", "2*Z11*Z12/(Z11+Z12)")):
        ws[f"Y{n}"], ws[f"Z{n}"] = name, f"=IFERROR({f},0)"
        ws[f"Z{n}"].number_format = "0.0000"

    # --- グラフ（縦軸は確率）---
    ch = BarChart()
    ch.type, ch.grouping = "col", "clustered"
    ch.title = "類似度の分布（各群内の相対度数）"
    ch.x_axis.title, ch.y_axis.title = "類似度（ビン下限）", "割合"
    ch.legend.position = "b"
    ch.width, ch.height = 15, 7.5
    ch.add_data(Reference(ws, min_col=13, max_col=14, min_row=1, max_row=len(bins) + 1),
                titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=11, min_row=2, max_row=len(bins) + 1))
    ws.add_chart(ch, "O6")

    for col, w in {"A": 28, "B": 46, "C": 46, "D": 10, "E": 10,
                   "F": 12, "G": 18, "H": 14}.items():
        ws.column_dimensions[col].width = w


def sheet3(wb, rows):
    """類似度行列。対角（自分自身）は空欄。"""
    ws = wb.create_sheet("Sheet3")
    names = sorted({label(r["image_a"]) for r in rows} | {label(r["image_b"]) for r in rows})
    idx = {name: i for i, name in enumerate(names)}
    for i, name in enumerate(names):
        ws.cell(row=1, column=i + 2, value=name)
        ws.cell(row=i + 2, column=1, value=name)
    for r in rows:
        a, b = idx[label(r["image_a"])], idx[label(r["image_b"])]
        for y, x in ((a, b), (b, a)):
            c = ws.cell(row=y + 2, column=x + 2, value=round(r["score"], 4))
            c.number_format = "0.0000"
    ws.column_dimensions["A"].width = 22
    for i in range(len(names)):
        ws.column_dimensions[get_column_letter(i + 2)].width = 10
    ws.freeze_panes = "B2"


def sheet2(wb, rows):
    """実験条件。META をそのまま書き出し、CSV から分かる項目を足す。"""
    ws = wb.create_sheet("Sheet2")
    same = sum(1 for r in rows if r["same_person"])
    items = META + [
        ("判定閾値", THRESHOLD),
        ("ペア数", f"{len(rows)}組（同一人物 {same} / 別人 {len(rows) - same}）"),
        ("結果CSV", os.path.basename(CSV)),
        ("集計方法", "tools/build_eval_report.py で自動生成"),
    ]
    for i, (k, v) in enumerate(items, start=1):
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)   # ラベルと値は必ず同じ行に置く
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 72
    for row in ws.iter_rows(min_col=2, max_col=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")


def main():
    if os.path.exists(OUT):
        # 実験IDの書き換え忘れで前回の成果物を壊さないための保険
        raise SystemExit(f"出力先に既にファイルがあります: {OUT}\n"
                         f"設定の OUT を確認するか、不要なら削除してください。")

    rows = read_rows(CSV)
    wb = Workbook()
    wb.remove(wb.active)
    sheet1(wb, rows)
    if BUILD_MATRIX:
        sheet3(wb, rows)
    sheet2(wb, rows)
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    wb.save(OUT)

    same = sum(1 for r in rows if r["same_person"])
    print(f"入力  : {CSV}")
    print(f"ペア数: {len(rows)}（同一人物 {same} / 別人 {len(rows) - same}）")
    print(f"出力  : {OUT}")


if __name__ == "__main__":
    main()
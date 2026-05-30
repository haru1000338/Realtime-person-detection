import csv
import os
from datetime import datetime

class DataLogger:
    def __init__(self, filepath="dwell_log.csv"):
        self.filepath = filepath
        
        # ファイルが存在しない場合（初回起動時）は、新仕様のヘッダーを作成する
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 🌟 NEW: 記録する項目を大幅に拡張
                writer.writerow(["Timestamp", "Booth_name", "Track_ID", "Real_ID", "Status", "ReID_Score", "Dwell_Time_sec"])
            print(f"📝 新規ログファイルを作成しました（拡張仕様）: {self.filepath}")

    # 🌟 NEW: 引数に real_id, status, reid_score を追加（エラー防止のため初期値を設定）
    def record_exit(self, track_id, dwell_time, booth_name, real_id="Unknown", status="Unknown", reid_score=0.0):
        """
        人がエリアから出た瞬間に呼び出され、詳細なAIデータをCSVに1行追記する関数
        """
        # 1秒未満の滞在（ただ通り過ぎただけのノイズなど）は記録しない
        if dwell_time < 1.0:
            return

        # 現在の日時を取得 (例: 2026-05-31 14:30:15)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # スコアの表記ゆれを防ぐ（float型以外が来ても0.00にする）
        try:
            formatted_score = f"{float(reid_score):.2f}"
        except (ValueError, TypeError):
            formatted_score = "0.00"

        # CSVファイルに追記モード('a')で書き込む
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, booth_name, track_id, real_id, status, formatted_score, f"{dwell_time:.1f}"])
        
        # ターミナル表示をアップグレード（分析結果も同時出力！）
        print(f"💾 [LOG] {booth_name}退場 | Track:{track_id} -> Real:{real_id} ({status}) | 類似度:{formatted_score} | 滞在:{dwell_time:.1f}秒")
import csv
import os
from datetime import datetime

class DataLogger:
    def __init__(self, filepath="dwell_log.csv"):
        self.filepath = filepath
        
        # ファイルが存在しない場合（初回起動時）は、ヘッダー（見出し）を作成する
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp","Booth_name" , "Track_ID", "Dwell_Time_sec"])
            print(f"📝 新規ログファイルを作成しました: {self.filepath}")

    def record_exit(self, track_id, dwell_time, booth_name):
        """
        人がエリアから出た瞬間に呼び出され、CSVにデータを1行追記する関数
        """
        # 1秒未満の滞在（ただ通り過ぎただけのノイズなど）は記録しない
        if dwell_time < 1.0:
            return

        # 現在の日時を取得 (例: 2026-05-09 14:30:15)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # CSVファイルに追記モード('a')で書き込む
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, booth_name, track_id, f"{dwell_time:.1f}"])
        
        # ターミナルにも状況を表示する（動いている感を演出！）
        print(f"💾 [LOG] {booth_name}から ID:{track_id} が退出しました。滞在時間: {dwell_time:.1f}秒")
import cv2
import numpy as np


class Visualizer:
    def draw(self, img, booths, tracks):
        output = img.copy()

        # ブースの枠線を描画
        for booth_name, pts in booths.items():
            cv2.polylines(output, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            text_x, text_y = pts[0][0], pts[0][1]
            cv2.putText(output, booth_name, (text_x + 5, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        for track in tracks:
            x0, y0, x1, y1 = track["box"]
            current_booth = track.get("current_booth")
            dwell_time = track.get("dwell_time", 0.0)
            label = track.get("label", f"ID:{track['track_id']}")
            status = track.get("real_status", "")
            trajectory_points = track.get("trajectory_points", [])
            
            # 🟢 Sで始まるか（スタッフか）判定するために real_id を取得
            real_id = track.get("real_id", "")

            # 🟢 色と太さの決定ロジック
            if real_id.startswith("S"):
                color = (255, 100, 0)  # スタッフ：明るい青色 (B, G, R)
                thickness = 3          # スタッフは枠線を太くして目立たせる
            else:
                color = (0, 0, 255) if current_booth else (0, 255, 0)  # 来場者：ブース内なら赤、外なら緑
                thickness = 2

            # 軌跡（歩いたルート）の描画
            if len(trajectory_points) >= 2:
                points = np.array(trajectory_points, dtype=np.int32).reshape(-1, 1, 2)
                # 軌跡の色も同じように合わせる
                cv2.polylines(output, [points], isClosed=False, color=color, thickness=2)

            # テキストラベルの生成
            text = label
            # if status:
            #     text += f" {status}"
            if current_booth and not real_id.startswith("S"):
                text += f" {dwell_time:.1f}sec"

            # 枠と文字の描画
            cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness)
            cv2.putText(output, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)

        return output
import cv2
import numpy as np


class Visualizer:
    def draw(self, img, booths, tracks):
        output = img.copy()

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

            color = (0, 0, 255) if current_booth else (0, 255, 0)

            if len(trajectory_points) >= 2:
                points = np.array(trajectory_points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(output, [points], isClosed=False, color=(0, 0, 255), thickness=3)

            text = label
            # if status:
            #     text += f" {status}"
            if current_booth:
                text += f" {dwell_time:.1f}sec"

            cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
            cv2.putText(output, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return output
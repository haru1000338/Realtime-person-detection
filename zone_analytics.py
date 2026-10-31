from collections import defaultdict
import time
import cv2
import numpy as np

class ZoneAnalytics:
    def __init__(self, buffer_time=3.0):
        self.buffer_time = buffer_time
        self.track_history = defaultdict(list)
        self.active_trackers = {}
        self.exit_candidates = {}
        self.booths_rate = {
            "Booth_A": np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], np.float32),
            "Booth_B": np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]], np.float32),
        }

    def build_booths(self, img_w, img_h):
        booths = {}
        for booth_name, rate_pts in self.booths_rate.items():
            pixel_pts = [[int(x * img_w), int(y * img_h)] for x, y in rate_pts]
            booths[booth_name] = np.array(pixel_pts, dtype=np.int32)
        return booths

    def _get_current_booth(self, foot_x, foot_y, booths):
        for booth_name, pts in booths.items():
            is_inside = cv2.pointPolygonTest(pts, (foot_x, foot_y), False)
            if is_inside >= 0:
                return booth_name
        return None

    def _record(self, data_logger, entry, end_time):
        data_logger.record_exit(entry["track_id"], end_time - entry["entry_time"], entry["Booth_name"],
                                entry["real_id"], entry["status"], entry["reid_score"])

    def update(self, tracks, frame_shape, data_logger):
        img_h, img_w = frame_shape[:2]
        booths = self.build_booths(img_w, img_h)
        current_time = time.time()
        current_ids_in_roi = set()
        enriched_tracks = []

        for track in tracks:
            track_id = track["track_id"]
            foot_x, foot_y = track["foot_point"]

            # 🌟 filter.py から渡された AIの判定結果（Real_ID等）を受け取る
            real_id = track.get("real_id", "Unknown")
            status = track.get("status", "Unknown")
            reid_score = track.get("reid_score", 0.0)

            # 軌跡は描画用なので track_id 単位で保持する
            self.track_history[track_id].append((foot_x, foot_y))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            current_booth = self._get_current_booth(foot_x, foot_y, booths)
            dwell_time = 0.0

            # 🌟 滞在は real_id 単位で管理する。照合前（waiting/Unknown）は計測しない
            if status not in ("waiting", "Unknown"):
                current_ids_in_roi.add(real_id)
                entry = {"Booth_name": current_booth, "entry_time": current_time, "track_id": track_id,
                         "real_id": real_id, "status": status, "reid_score": reid_score}

                if current_booth:
                    if real_id in self.active_trackers:
                        previous = self.active_trackers[real_id]
                        if previous["Booth_name"] != current_booth:
                            # ブースを移動した瞬間の記録！
                            self._record(data_logger, previous, current_time)
                            self.active_trackers[real_id] = entry
                        else:
                            # 同じブースに滞在中（別トラックで戻ってきた場合もここで継続）
                            previous.update(track_id=track_id, status=status, reid_score=reid_score)
                    else:
                        # 初めてブースに入った人
                        self.active_trackers[real_id] = entry

                    # ロスト中に戻ってきた場合は滞在を継続
                    self.exit_candidates.pop(real_id, None)
                    dwell_time = current_time - self.active_trackers[real_id]["entry_time"]
                elif real_id in self.active_trackers:
                    # 画面内でブースの外に出た瞬間の記録
                    self._record(data_logger, self.active_trackers.pop(real_id), current_time)
                    self.exit_candidates.pop(real_id, None)

            track["current_booth"] = current_booth
            track["dwell_time"] = dwell_time
            track["trajectory_points"] = list(self.track_history[track_id])
            enriched_tracks.append(track)

        # 画面から消えた（ロストした）人の処理
        for real_id in self.active_trackers:
            if real_id not in current_ids_in_roi and real_id not in self.exit_candidates:
                self.exit_candidates[real_id] = current_time

        # バッファ時間を過ぎて「完全に退出した」とみなされた人の最終記録！
        for real_id, lost_time in list(self.exit_candidates.items()):
            if current_time - lost_time > self.buffer_time:
                # 見失った時点までを滞在とする（buffer_time 分は含めない）
                self._record(data_logger, self.active_trackers.pop(real_id), lost_time)
                del self.exit_candidates[real_id]

        return enriched_tracks, booths
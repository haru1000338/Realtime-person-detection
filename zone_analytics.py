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

    def update(self, tracks, frame_shape, data_logger):
        img_h, img_w = frame_shape[:2]
        booths = self.build_booths(img_w, img_h)
        current_time = time.time()
        current_ids_in_roi = set()
        enriched_tracks = []

        for track in tracks:
            track_id = track["track_id"]
            foot_x, foot_y = track["foot_point"]
            
            # 🌟 NEW: filter.py から渡された AIの最新判定結果（Real_ID等）を受け取る
            real_id = track.get("real_id", "Unknown")
            status = track.get("status", "Unknown")
            reid_score = track.get("reid_score", 0.0)

            current_ids_in_roi.add(track_id)

            self.track_history[track_id].append((foot_x, foot_y))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            current_booth = self._get_current_booth(foot_x, foot_y, booths)
            dwell_time = 0.0

            if current_booth:
                if track_id in self.active_trackers:
                    previous_booth = self.active_trackers[track_id]["Booth_name"]
                    
                    # ブースを移動した瞬間の記録！
                    if previous_booth != current_booth:
                        dwell_time = current_time - self.active_trackers[track_id]["entry_time"]
                        
                        # 🌟 退出時の「最終確定したAIデータ」を取り出してロガーに渡す
                        past_real_id = self.active_trackers[track_id].get("real_id", "Unknown")
                        past_status = self.active_trackers[track_id].get("status", "Unknown")
                        past_score = self.active_trackers[track_id].get("reid_score", 0.0)
                        
                        data_logger.record_exit(track_id, dwell_time, previous_booth, past_real_id, past_status, past_score)
                        
                        self.active_trackers[track_id] = {
                            "Booth_name": current_booth,
                            "entry_time": current_time,
                            "real_id": real_id,
                            "status": status,
                            "reid_score": reid_score
                        }
                    else:
                        # 🌟 ここが超重要！
                        # 同じブースに滞在している間にも、AIが「仮ID」から「S001」に動的昇格する可能性があるため、常に最新情報で上書き更新し続ける
                        self.active_trackers[track_id]["real_id"] = real_id
                        self.active_trackers[track_id]["status"] = status
                        self.active_trackers[track_id]["reid_score"] = reid_score
                else:
                    # 初めてブースに入った人
                    self.active_trackers[track_id] = {
                        "Booth_name": current_booth,
                        "entry_time": current_time,
                        "real_id": real_id,
                        "status": status,
                        "reid_score": reid_score
                    }

                if track_id in self.exit_candidates:
                    del self.exit_candidates[track_id]

                dwell_time = current_time - self.active_trackers[track_id]["entry_time"]

            track["current_booth"] = current_booth
            track["dwell_time"] = dwell_time
            track["trajectory_points"] = list(self.track_history[track_id])
            enriched_tracks.append(track)

        # 画面から消えた（ロストした）人の処理
        for track_id in list(self.active_trackers.keys()):
            if track_id not in current_ids_in_roi and track_id not in self.exit_candidates:
                self.exit_candidates[track_id] = {
                    "booth_name": self.active_trackers[track_id]["Booth_name"],
                    "entry_time": self.active_trackers[track_id]["entry_time"],
                    "lost_time": current_time,
                    # 🌟 画面から消える直前の「最も精度の高い状態」をコピーして退避させておく
                    "real_id": self.active_trackers[track_id].get("real_id", "Unknown"),
                    "status": self.active_trackers[track_id].get("status", "Unknown"),
                    "reid_score": self.active_trackers[track_id].get("reid_score", 0.0)
                }

        # バッファ時間を過ぎて「完全に退出した」とみなされた人の最終記録！
        for track_id in list(self.exit_candidates.keys()):
            lost_duration = current_time - self.exit_candidates[track_id]["lost_time"]
            if lost_duration > self.buffer_time:
                booth_name = self.exit_candidates[track_id]["booth_name"]
                final_dwell_time = current_time - self.exit_candidates[track_id]["entry_time"]
                
                # 🌟 退避させておいた最終確定データを取り出してロガーに渡す
                final_real_id = self.exit_candidates[track_id].get("real_id", "Unknown")
                final_status = self.exit_candidates[track_id].get("status", "Unknown")
                final_score = self.exit_candidates[track_id].get("reid_score", 0.0)

                data_logger.record_exit(track_id, final_dwell_time, booth_name, final_real_id, final_status, final_score)

                if track_id in self.active_trackers:
                    del self.active_trackers[track_id]
                del self.exit_candidates[track_id]

        return enriched_tracks, booths
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
            current_ids_in_roi.add(track_id)

            self.track_history[track_id].append((foot_x, foot_y))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            current_booth = self._get_current_booth(foot_x, foot_y, booths)
            dwell_time = 0.0

            if current_booth:
                if track_id in self.active_trackers:
                    previous_booth = self.active_trackers[track_id]["Booth_name"]
                    if previous_booth != current_booth:
                        dwell_time = current_time - self.active_trackers[track_id]["entry_time"]
                        data_logger.record_exit(track_id, dwell_time, previous_booth)
                        self.active_trackers[track_id] = {
                            "Booth_name": current_booth,
                            "entry_time": current_time,
                        }
                else:
                    self.active_trackers[track_id] = {
                        "Booth_name": current_booth,
                        "entry_time": current_time,
                    }

                if track_id in self.exit_candidates:
                    del self.exit_candidates[track_id]

                dwell_time = current_time - self.active_trackers[track_id]["entry_time"]

            track["current_booth"] = current_booth
            track["dwell_time"] = dwell_time
            track["trajectory_points"] = list(self.track_history[track_id])
            enriched_tracks.append(track)

        for track_id in list(self.active_trackers.keys()):
            if track_id not in current_ids_in_roi and track_id not in self.exit_candidates:
                self.exit_candidates[track_id] = {
                    "booth_name": self.active_trackers[track_id]["Booth_name"],
                    "entry_time": self.active_trackers[track_id]["entry_time"],
                    "lost_time": current_time,
                }

        for track_id in list(self.exit_candidates.keys()):
            lost_duration = current_time - self.exit_candidates[track_id]["lost_time"]
            if lost_duration > self.buffer_time:
                booth_name = self.exit_candidates[track_id]["booth_name"]
                final_dwell_time = current_time - self.exit_candidates[track_id]["entry_time"]
                data_logger.record_exit(track_id, final_dwell_time, booth_name)

                if track_id in self.active_trackers:
                    del self.active_trackers[track_id]
                del self.exit_candidates[track_id]

        return enriched_tracks, booths
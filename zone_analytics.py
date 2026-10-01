from collections import defaultdict
import time
import cv2
import numpy as np

class ZoneAnalytics:
    def __init__(self, buffer_time=3.0):
        """
        buffer_time: 猶予時間
        track_history: 各トラックIDの足元座標の履歴を保持する辞書
        active_trackers: 現在アクティブな人の情報を保持する辞書
        exit_candidates: 退出候補の人の辞書
        """
        self.buffer_time = buffer_time
        self.track_history = defaultdict(list)
        self.active_trackers = {}
        self.exit_candidates = {}
        self.booths_rate = {
            "Booth_A": np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], np.float32),
            "Booth_B": np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]], np.float32),
        }

    def build_booths(self, img_w, img_h):
        """
        ブースの座標を画像サイズに基づいてピクセル単位に変換する
        img_w, img_h: カメラから届いた画像全体の幅と高さ
        return: keyがブース名、valueがそのブースのポリゴン座標の辞書
        """
        booths = {}
        for booth_name, rate_pts in self.booths_rate.items():
            pixel_pts = [[int(x * img_w), int(y * img_h)] for x, y in rate_pts]
            booths[booth_name] = np.array(pixel_pts, dtype=np.int32)
        return booths

    def _get_current_booth(self, foot_x, foot_y, booths):
        """
        指定された座標がどのブース内にあるかを判定する
        foot_x, foot_y: 人の足元の座標
        booths: ブースの座標の辞書
        return: その人が属するブースの名前、またはNone
        """
        for booth_name, pts in booths.items():
            # cv2.pointPolygonTestを使って、(foot_x, foot_y) がブースのポリゴン内にあるかどうかを判定する
            # Falseは距離を返さず、内外判定のみ(inside:1, on edge:0, outside:-1)
            is_inside = cv2.pointPolygonTest(pts, (foot_x, foot_y), False) 
            if is_inside >= 0:
                return booth_name
        return None

    def update(self, tracks, frame_shape, data_logger):
        """
        現在のフレームの情報を更新し、ブース情報を付加したトラックを返す

        tracks: 現在のフレームで検出されたトラックのリスト
        frame_shape: 現在のフレームの形状 (height, width, channels)
        data_logger: 退出時の記録を行うロガーオブジェクト
        return: (enriched_tracks, booths)
                enriched_tracks - 各トラックに current_booth / dwell_time /
                                trajectory_points を付加したリスト
                booths - ピクセル単位のブース座標の辞書
        """
        img_h, img_w = frame_shape[:2]
        booths = self.build_booths(img_w, img_h)
        current_time = time.time()
        current_ids_in_roi = set()
        enriched_tracks = []

        # 一人ずつの処理
        for track in tracks:
            track_id = track["track_id"]
            foot_x, foot_y = track["foot_point"]
            
            # filter.py から渡された AIの最新判定結果（Real_ID等）を受け取る
            # trackはdict型
            real_id = track.get("real_id", "Unknown")
            status = track.get("status", "Unknown")
            reid_score = track.get("reid_score", 0.0)

            current_ids_in_roi.add(track_id)

            # 上限30件で足元座標履歴を保持
            self.track_history[track_id].append((foot_x, foot_y))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            current_booth = self._get_current_booth(foot_x, foot_y, booths)
            dwell_time = 0.0

            # 足元がブース内にある場合
            if current_booth:
                if track_id in self.active_trackers:
                    previous_booth = self.active_trackers[track_id]["Booth_name"]
                    
                    # ブースを移動した瞬間の記録
                    if previous_booth != current_booth:
                        # TODO: dwell_timeがどこで使われるかを確認する
                        # このdwell_timeは「移動前のブースでの滞在時間」を計算するために使用する
                        # 直後に entry_timeをcurrent_timeに更新する→ブロック末尾で再計算される値がenriched_tracksに格納される
                        dwell_time = current_time - self.active_trackers[track_id]["entry_time"]
                        
                        # ブースを移動する前の情報をロガーに記録する
                        past_real_id = self.active_trackers[track_id].get("real_id", "Unknown")
                        past_status = self.active_trackers[track_id].get("status", "Unknown")
                        past_score = self.active_trackers[track_id].get("reid_score", 0.0)
                        
                        # 退出情報をロガーに記録
                        data_logger.record_exit(track_id, dwell_time, previous_booth, past_real_id, past_status, past_score)
                        
                        self.active_trackers[track_id] = {
                            "Booth_name": current_booth,
                            "entry_time": current_time,
                            "real_id": real_id,
                            "status": status,
                            "reid_score": reid_score
                        }
                    else:
                        # 同じブースに滞在している間にも、判定結果が更新される可能性があるので、最新の情報を保持する
                        self.active_trackers[track_id]["real_id"] = real_id
                        self.active_trackers[track_id]["status"] = status
                        self.active_trackers[track_id]["reid_score"] = reid_score
                else:
                    # 初めてブースに入った人の情報を記録する
                    self.active_trackers[track_id] = {
                        "Booth_name": current_booth,
                        "entry_time": current_time,
                        "real_id": real_id,
                        "status": status,
                        "reid_score": reid_score
                    }
                # ブース内にいる間は、退出候補リストから削除する
                if track_id in self.exit_candidates:
                    del self.exit_candidates[track_id]

                dwell_time = current_time - self.active_trackers[track_id]["entry_time"]

            # 情報を追加して enriched_tracks に格納
            track["current_booth"] = current_booth
            track["dwell_time"] = dwell_time
            track["trajectory_points"] = list(self.track_history[track_id])
            enriched_tracks.append(track)

        # 画面から消えた（ロストした）人の処理
        for track_id in list(self.active_trackers.keys()):
            # 条件: 現在のフレームに存在しないIDで、かつ exit_candidates にも存在しない場合
            if track_id not in current_ids_in_roi and track_id not in self.exit_candidates:
                # 画面から消えた人の情報を exit_candidates に追加
                self.exit_candidates[track_id] = {
                    "booth_name": self.active_trackers[track_id]["Booth_name"],
                    "entry_time": self.active_trackers[track_id]["entry_time"],
                    "lost_time": current_time,
                    # 画面から消える直前の「最も精度の高い状態」をコピーして退避させておく
                    # （最大スコアではなく、最後に取得したスコアを保持する）
                    "real_id": self.active_trackers[track_id].get("real_id", "Unknown"),
                    "status": self.active_trackers[track_id].get("status", "Unknown"),
                    "reid_score": self.active_trackers[track_id].get("reid_score", 0.0)
                }

        # バッファ時間を過ぎて「完全に退出した」とみなされた人の最終記録
        for track_id in list(self.exit_candidates.keys()):
            lost_duration = current_time - self.exit_candidates[track_id]["lost_time"]
            if lost_duration > self.buffer_time:
                booth_name = self.exit_candidates[track_id]["booth_name"]
                final_dwell_time = current_time - self.exit_candidates[track_id]["entry_time"]
                
                # 最終的な判定結果を取得する
                final_real_id = self.exit_candidates[track_id].get("real_id", "Unknown")
                final_status = self.exit_candidates[track_id].get("status", "Unknown")
                final_score = self.exit_candidates[track_id].get("reid_score", 0.0)
                
                # ロガーに最終記録を残す
                data_logger.record_exit(track_id, final_dwell_time, booth_name, final_real_id, final_status, final_score)

                # 退出候補リストとアクティブトラッカーから削除する
                if track_id in self.active_trackers:
                    del self.active_trackers[track_id]
                del self.exit_candidates[track_id]

        return enriched_tracks, booths
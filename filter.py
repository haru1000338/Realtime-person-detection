import cv2
import numpy as np
from collections import defaultdict
import time

import reid

track_history = defaultdict(list)  # トラッキングIDごとの足跡を保存する辞書
active_trackers = {}  # { ID: {Booth_name: str}, 'entry_time': float } }
exit_candidates = {}  # { ID: {booth_name: str, exit_time: float} } 
BUFFER_TIME = 3.0  # 退出と判断するまでの猶予時間（秒）

person_features = {}
person_roles = {}

# ({左上}, {右上}, {右下}, {左下}) の順でブースのポリゴンを定義(範囲：0~1)
BOOTHS_RATE = {
    "Booth_A": np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], np.float32),  # 左上のブース
    "Booth_B": np.array([[0.5, 0], [1, 0], [1, 1], [0.5, 1]], np.float32)   # 右下のブース
}

def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    """コントラストと明るさを調整"""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def process_frame(model, img, heatmap_generator, data_logger, conf_threshold=0.5, show_heatmap=True):
    """1フレームの画像を受け取り、追跡（トラッキング）と描画を行う"""
    img = adjust_contrast_brightness(img, contrast=1.0, brightness=0)
    img_h, img_w = img.shape[:2]

    BOOTHS = {}

    for booth_name, rate_pts in BOOTHS_RATE.items():
        pixcel_pts = [[int(x * img_w), int(y * img_h)] for x, y in rate_pts]
        BOOTHS[booth_name] = np.array(pixcel_pts, dtype=np.int32)

    # 【魔法の1行】ただの推論ではなく、trackモードでByteTrack（ID追跡）を有効にする
    results = model.track(img, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
    processed_results = []
    current_foot_positions = []  # 現在のフレームでの足の位置を保存するリスト

    current_ids_in_roi = set()  # 現在ROI内にいるIDの集合


    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)  # トラッキングIDを取得
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):
            if int(cls) == 0:  # personクラスのみ
                current_ids_in_roi.add(track_id)
                x0, y0, x1, y1 = map(int, box)
                foot_x = int((x0 + x1) / 2)
                foot_y = int(y1)
                current_foot_positions.append((foot_x, foot_y))
    
    # ヒートマップは常に内部で更新するが、表示は `show_heatmap` によって制御する
    img = heatmap_generator.apply(img, current_foot_positions, show=show_heatmap)

    for booth_name, pts in BOOTHS.items():
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        text_x, text_y = pts[0][0], pts[0][1]
        cv2.putText(img, booth_name, (text_x + 5, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 誰かが検出され、かつIDが割り振られている場合のみ処理
    if results[0].boxes.id is not None:
        scores = results[0].boxes.conf.cpu().numpy()

        for box, score, track_id, cls in zip(boxes, scores, ids, classes):
            if int(cls) == 0:  # personクラスのみ
                x0, y0, x1, y1 = map(int, box)

                # 特徴を抽出→IDと特徴を紐づけて保存
                if track_id not in person_features:
                    crop_y0, crop_y1 = max(0, y0), min(img_h, y1)
                    crop_x0, crop_x1 = max(0, x0), min(img_w, x1)
                    crop_img = img[crop_y0:crop_y1, crop_x0:crop_x1]

                    if crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
                        new_feature = reid.get_feature(crop_img)

                        best_match_id = None
                        heighest_score = 0.0

                        for past_id, past_feature in person_features.items():
                            if past_feature is not None:
                                similarity = reid.compare_features(new_feature, past_feature)
                                if similarity > heighest_score:
                                    heighest_score = similarity
                                    best_match_id = past_id

                        THRESHOLD = 0.7
                        if heighest_score > THRESHOLD:
                            print(f"[Re-ID] ID:{track_id} が過去の ID:{best_match_id} と類似度 {heighest_score:.2f} でマッチ！")
                            person_roles[track_id] = f"Same as ID:{best_match_id}"
                        else:
                            print(f"[Re-ID] ID:{track_id} は新規の人物と判断 (最高類似度 {heighest_score:.2f})")
                            person_roles[track_id] = f"New Person"
                        
                        person_features[track_id] = new_feature
                    else:
                        person_features[track_id] = None  # 特徴抽出できない場合はNoneを保存
                        person_roles[track_id] = "Unknown"
                
                processed_results.append((x0, y0, x1, y1, score, track_id))

                foot_x = int((x0 + x1) / 2)
                foot_y = int(y1)

                track_history[track_id].append((foot_x, foot_y))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                points = np.array(track_history[track_id], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [points], isClosed=False, color=(0, 0, 255), thickness=3)

                current_booth = None
                for booth_name, pts in BOOTHS.items():
                    is_inside = cv2.pointPolygonTest(pts, (foot_x, foot_y), False)
                    if is_inside >= 0:
                        current_booth = booth_name
                        break

                dwell_time = 0.0

                if current_booth:
                    # 🌟 置き換えスタート：ブース移動と新規入場の判定
                    if track_id in active_trackers:
                        previous_booth = active_trackers[track_id]['Booth_name']
                        
                        # パターンA：別のブースへ移動した時！
                        if previous_booth != current_booth:
                            dwell_time = time.time() - active_trackers[track_id]['entry_time']
                            print(f"🔄 [DEBUG] ID:{track_id} が {previous_booth} から {current_booth} へ移動！ (滞在: {dwell_time:.1f}秒)")
                            
                            # 🚨 ここで忘れずに古いブースの記録をロガーに渡す！
                            data_logger.record_exit(track_id, dwell_time, previous_booth)
                            
                            # 新しいブースの情報で上書き
                            active_trackers[track_id] = {
                                'Booth_name': current_booth, 
                                'entry_time': time.time()
                            }
                    else:
                        # パターンB：新規入場
                        print(f"🆕 [DEBUG] ID:{track_id} が {current_booth} に入りました！")
                        active_trackers[track_id] = {
                            'Booth_name': current_booth, 
                            'entry_time': time.time()
                        }
                    # 🌟 置き換えここまで
                    
                    
                    if track_id in exit_candidates:
                        del exit_candidates[track_id]

                    dwell_time = time.time() - active_trackers[track_id]['entry_time']

                # 画像に枠と「ID」を描画
                color = (0, 0, 255) if current_booth else (0, 255, 0)
                role = person_roles.get(track_id, "")
                text = f'ID:{track_id} {role}'

                if current_booth:
                    text += f'{current_booth} {dwell_time:.1f}sec'


                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                cv2.putText(img, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# ==========================================
#  退出処理（バッファ付きクリーンアップ）
# ==========================================
    current_time = time.time()

    for track_id in list(active_trackers.keys()):
        if track_id not in current_ids_in_roi:
            if track_id not in exit_candidates:
                exit_candidates[track_id] = {
                    'booth_name': active_trackers[track_id]['Booth_name'],
                    'entry_time': active_trackers[track_id]['entry_time'],
                    'lost_time': current_time
                }
                
    for track_id in list(exit_candidates.keys()):
        lost_duration = current_time - exit_candidates[track_id]['lost_time']

        if lost_duration > BUFFER_TIME:
            booth_name = exit_candidates[track_id]['booth_name']
            final_dwell_time = current_time - exit_candidates[track_id]['entry_time']

            data_logger.record_exit(track_id, final_dwell_time, booth_name)

            if track_id in active_trackers:
                del active_trackers[track_id]
            del exit_candidates[track_id]

    return img, processed_results
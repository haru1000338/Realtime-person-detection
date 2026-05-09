import cv2
import numpy as np
from collections import defaultdict
import time

track_history = defaultdict(list)  # トラッキングIDごとの足跡を保存する辞書
entry_times = {}  # トラッキングIDごとの入場時間を保存する辞書

def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    """コントラストと明るさを調整"""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

def process_frame(model, img, heatmap_generator, data_logger, conf_threshold=0.5):
    """1フレームの画像を受け取り、追跡（トラッキング）と描画を行う"""
    img = adjust_contrast_brightness(img, contrast=1.0, brightness=0)

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
                x0, y0, x1, y1 = map(int, box)
                foot_x = int((x0 + x1) / 2)
                foot_y = int(y1)
                current_foot_positions.append((foot_x, foot_y))
    
    img = heatmap_generator.apply(img, current_foot_positions)

    roi_x1, roi_y1 = 0, int(img.shape[0] * 0.5)
    roi_x2, roi_y2 = img.shape[1], img.shape[0]

    # 誰かが検出され、かつIDが割り振られている場合のみ処理
    if results[0].boxes.id is not None:
        scores = results[0].boxes.conf.cpu().numpy()

        for box, score, track_id, cls in zip(boxes, scores, ids, classes):
            if int(cls) == 0:  # personクラスのみ
                x0, y0, x1, y1 = map(int, box)
                
                # 結果に track_id も含めて返す
                processed_results.append((x0, y0, x1, y1, score, track_id))

                foot_x = int((x0 + x1) / 2)
                foot_y = int(y1)

                track_history[track_id].append((foot_x, foot_y))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                points = np.array(track_history[track_id], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [points], isClosed=False, color=(0, 0, 255), thickness=3)

                is_inside_roi = (roi_x1 <= foot_x <= roi_x2) and (roi_y1 <= foot_y <= roi_y2)
                dwell_time = 0.0

                if is_inside_roi:
                    current_ids_in_roi.add(track_id)

                    if track_id not in entry_times:
                        entry_times[track_id] = time.time()
                    dwell_time = time.time() - entry_times[track_id]

                # 画像に枠と「ID」を描画
                color = (0, 255, 0)
                text = f'ID:{track_id} ({score:.2f})'

                if is_inside_roi:
                    color = (0, 0, 255)
                    text += f' {dwell_time:.1f}sec'


                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                cv2.putText(img, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    lost_ids = list(set(entry_times.keys()) - current_ids_in_roi)

    for lost_id in lost_ids:
        final_dwell_time = time.time() - entry_times[lost_id]
        data_logger.record_exit(lost_id, final_dwell_time)

        del entry_times[lost_id]

    return img, processed_results
import cv2

from id_manager import IDManager
from visualizer import Visualizer
from zone_analytics import ZoneAnalytics

def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    """コントラストと明るさを調整"""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


id_manager = IDManager()
zone_analytics = ZoneAnalytics()
visualizer = Visualizer()

def process_frame(model, img, heatmap_generator, data_logger, conf_threshold=0.5, show_heatmap=True):
    """1フレームの画像を受け取り、追跡（トラッキング）と描画を行う"""
    source_img = adjust_contrast_brightness(img, contrast=1.0, brightness=0)

    # 【魔法の1行】ただの推論ではなく、trackモードでByteTrack（ID追跡）を有効にする
    results = model.track(source_img, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
    processed_results = []
    current_foot_positions = []  # 現在のフレームでの足の位置を保存するリスト
    raw_tracks = []
    
    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        for box, score, track_id, cls in zip(boxes, scores, ids, classes):
            if int(cls) != 0:
                continue

            x0, y0, x1, y1 = map(int, box)
            crop_y0, crop_y1 = max(0, y0), min(source_img.shape[0], y1)
            crop_x0, crop_x1 = max(0, x0), min(source_img.shape[1], x1)
            crop_img = source_img[crop_y0:crop_y1, crop_x0:crop_x1]

            # 🌟 IDマネージャーの判定（新ロジック）を実行！
            id_result = id_manager.resolve(track_id, crop_img)
            
            foot_x = int((x0 + x1) / 2)
            foot_y = int(y1)

            current_foot_positions.append((foot_x, foot_y))
            processed_results.append((x0, y0, x1, y1, score, track_id))

            # 🌟 NEW: CSVに書き込むためのデータを整形して抽出
            reid_score_val = 0.0
            status_str = id_result.status
            
            # id_managerが "matched:0.85" のように返してきた場合、数字と文字を分解する
            if "matched:" in status_str:
                try:
                    reid_score_val = float(status_str.split(":")[1])
                except ValueError:
                    reid_score_val = 0.0
                status_str = "matched_visitor"
            elif status_str == "staff" or status_str == "waiting" or status_str == "new_visitor":
                # このパターンのスコアは一旦0（またはログで確認不要）とする
                reid_score_val = 0.0 

            # raw_tracksの辞書にAIの判定結果をフルセットで詰め込む！
            raw_tracks.append({
                "track_id": track_id,
                "real_id": id_result.real_id,          # CSV用
                "status": status_str,                  # CSV用（整形済み）
                "reid_score": reid_score_val,          # CSV用
                "real_status": id_result.status,       # visualizer描画用（元のまま）
                "label": id_result.label,              # visualizer描画用
                "box": (x0, y0, x1, y1),
                "score": float(score),
                "foot_point": (foot_x, foot_y),
            })

    # ここで raw_tracks を zone_analytics に渡す。この中で logger に命令が飛ぶ！
    annotated_tracks, booths = zone_analytics.update(raw_tracks, source_img.shape, data_logger)
    annotated_img = heatmap_generator.apply(source_img, current_foot_positions, show=show_heatmap)
    annotated_img = visualizer.draw(annotated_img, booths, annotated_tracks)

    return annotated_img, processed_results
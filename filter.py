import cv2

from id_manager import IDManager
from visualizer import Visualizer
from zone_analytics import ZoneAnalytics

def adjust_contrast_brightness(img, contrast=1.0, brightness=0):
    """
    コントラストと明るさを調整
    alpha: コントラストの係数 (1.0で元のコントラスト)
    beta: 明るさの係数 (0で元の明るさ)
    """
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

# 他のファイルからインポートされたクラスのインスタンスを作成
# TODO: これらのインスタンスはグローバルに作成するのではなく、必要な関数内で作成する方が良いかも？-> 現状：importしただけで.ptを読み込む
id_manager = IDManager()
zone_analytics = ZoneAnalytics()
visualizer = Visualizer()

def process_frame(model, img, heatmap_generator, data_logger, conf_threshold=0.5, show_heatmap=True):
    """1フレームの画像を受け取り、追跡（トラッキング）と描画を行う"""
    # 画像のコントラストと明るさを調整
    # TODO: 現状何も変えていない
    source_img = adjust_contrast_brightness(img, contrast=1.0, brightness=0)

    # ByteTrackを使ってトラッキングを実行
    # TODO: この時点で追跡対象を人限定にできる？→classes=[0]を指定することで人限定にできる
    results = model.track(source_img, conf=conf_threshold, persist=True, tracker="bytetrack.yaml", verbose=False)
    processed_results = []  # 追跡結果を保存するリスト
    current_foot_positions = []  # 現在のフレームでの足の位置を保存するリスト
    raw_tracks = []  # csv書き込み用データのリスト
    
    # 追跡対象となる人が存在する場合のみ処理を行う
    # BytetrackがIDを付与できなかった場合、検出はされているがIDがNoneとなるため、IDがNoneでない場合のみ処理する
    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 追跡対象のBBox座標を取得
        ids = results[0].boxes.id.cpu().numpy().astype(int)  # 追跡対象のIDを取得(ここで言うIDはtrack_id)
        classes = results[0].boxes.cls.cpu().numpy()  # 追跡対象のクラスを取得（人かどうかの判定に使用）
        scores = results[0].boxes.conf.cpu().numpy()  # 追跡対象の信頼度スコアを取得

        for box, score, track_id, cls in zip(boxes, scores, ids, classes):
            # ここでクラスが人（class=0）でない場合はスキップする
            if int(cls) != 0:
                continue

            # BBoxの左上(crop_x0, crop_y0)と右下(crop_x1, crop_y1)の座標を計算し、画像を切り抜く
            x0, y0, x1, y1 = map(int, box)
            crop_y0, crop_y1 = max(0, y0), min(source_img.shape[0], y1)
            crop_x0, crop_x1 = max(0, x0), min(source_img.shape[1], x1)
            crop_img = source_img[crop_y0:crop_y1, crop_x0:crop_x1]

            # 画像をid_managerに渡して、(real_id, status, label)が返る
            id_result = id_manager.resolve(track_id, crop_img)
            
            # 足元の座標を計算
            # TODO: x0とcrop_x0のどちらを使うかは要検討
            foot_x = int((x0 + x1) / 2)
            foot_y = int(y1)

            current_foot_positions.append((foot_x, foot_y))

            processed_results.append((x0, y0, x1, y1, score, track_id))


            reid_score_val = 0.0
            status_str = id_result.status
            
            # id_managerが "matched:0.85" のように返してきた場合、数字と文字を分解する
            # TODO: これほんとに必要か検討する（返り値の統一？）
            if "matched:" in status_str:
                try:
                    reid_score_val = float(status_str.split(":")[1])
                except ValueError:
                    reid_score_val = 0.0
                status_str = "matched_visitor"
            elif status_str == "staff" or status_str == "waiting" or status_str == "new_visitor":
                # このパターンのスコアは一旦0（またはログで確認不要）とする
                reid_score_val = 0.0 

            # raw_tracksの辞書にAIの判定結果をフルセットで詰め込む
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

    # zone_analytics, heatmap_generator, visualizerを使って、滞在時間の計算と描画を行う
    annotated_tracks, booths = zone_analytics.update(raw_tracks, source_img.shape, data_logger)
    # TODO: これだけインスタンスを作成していない理由
    annotated_img = heatmap_generator.apply(source_img, current_foot_positions, show=show_heatmap)
    annotated_img = visualizer.draw(annotated_img, booths, annotated_tracks)

    return annotated_img, processed_results
import cv2
import time
from ultralytics import YOLO

from filter import process_frame, id_manager, register_largest_as_staff
from heatmap import HeatmapGenerator
from logger import DataLogger

def main():
    # ============ 入力ソースの選択 ============
    # カメラを使う場合はこちらをコメント解除
    VIDEO_SOURCE = 0

    heatmap_generator = HeatmapGenerator()
    # データロガーの初期化
    data_logger = DataLogger()
    
    # 動画ファイルを使う場合はこちらをコメント解除
    # VIDEO_SOURCE = "test_movie/"  # testする動画ファイルを指定
    
    # 1. モデルとカメラの準備
    model = YOLO("yolo26n.pt")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("入力ソースを開けませんでした。")
        return
    

    target_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_ms = 1000.0 / target_fps if target_fps and target_fps > 0 else None
    prev_display_time = None

    show_metrics = False  # FPSや処理時間を表示するかどうか
    frame_size_printed = False  # フレームサイズ出力フラグ
    # ヒートマップ表示フラグ（`h` キーで切替）。裏では常にヒートマップを更新する。
    show_heatmap = True

    ret, frame = cap.read()
    if not ret:
        print("カメラ映像が取得できませんでした。")
        return
    height, width = frame.shape[:2]
    cv2.namedWindow("Webcam + Filter", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Webcam + Filter", width, height)
    print(f"📸 フレームサイズ: {width}x{height}")

    # ======
    print(cap.get(cv2.CAP_PROP_FPS))
    # ======

    try:
        while True:
            # 2. カメラから1フレームを読み込む
            frame_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("カメラ映像が取得できませんでした。")
                break

            capture_time = time.perf_counter()

            # 3. filter.py の処理結果をフレームに重ねる
            annotated_frame, results = process_frame(model, frame, heatmap_generator, data_logger, show_heatmap=show_heatmap)

            display_time = time.perf_counter()
            processing_ms = (display_time - capture_time) * 1000.0
            frame_ms = (display_time - frame_start) * 1000.0
            actual_fps = 1.0 / (display_time - prev_display_time) if prev_display_time else 0.0
            prev_display_time = display_time

            lag_ms = 0.0
            if target_frame_ms is not None:
                lag_ms = max(0.0, processing_ms - target_frame_ms)

            # 4. 検出件数とFPSを表示する
            overlay = annotated_frame.copy()
            if show_metrics:
                y = 30
                cv2.putText(overlay, f"Persons: {len(results)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                y += 40
                cv2.putText(overlay, f"Process: {processing_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y += 40
                cv2.putText(overlay, f"Loop: {frame_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                y += 40
                cv2.putText(overlay, f"FPS: {actual_fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y += 40
                if target_frame_ms is not None:
                    cv2.putText(overlay, f"Lag vs camera: {lag_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Webcam + Filter", overlay)

            # 5. 'q' で終了
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                show_metrics = not show_metrics
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                print(f"Heatmap display: {'ON' if show_heatmap else 'OFF'}")

            elif key == ord('s'):
                print("🔍 スタッフの特徴量を保存中...")
                # 追跡中で最も大きく映っている人物を、track_id ごと登録する
                new_staff_id = register_largest_as_staff(frame, results)
                if new_staff_id:
                    print(f"✅ スタッフID {new_staff_id} の特徴量を保存しました。")

                    flash_frame = overlay.copy()
                    cv2.rectangle(flash_frame, (0, 0), (flash_frame.shape[1], flash_frame.shape[0]), (0, 255, 0), -1)
                    cv2.imshow("Webcam + Filter", cv2.addWeighted(flash_frame, 0.5, overlay, 0.5, 0))
                    cv2.waitKey(100)
                else:
                    print("⚠️ スタッフの特徴量を保存できませんでした。顔がはっきり写っていることを確認してください。")
                    
    finally:
        id_manager.save_features()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
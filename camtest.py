import cv2
import time
from ultralytics import YOLO

from filter import process_frame

def main():
    # 1. モデルとカメラの準備
    model = YOLO("yolo26n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けませんでした。")
        return

    target_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_ms = 1000.0 / target_fps if target_fps and target_fps > 0 else None
    prev_display_time = None

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
            annotated_frame, results = process_frame(model, frame, conf_threshold=0.6)

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
            y = 30
            cv2.putText(overlay, f"Detections: {len(results)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
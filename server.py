import cv2
import uvicorn
import asyncio
import numpy as np
import time  # 🌟 NEW: FPS計測用に追加
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO

# AI推論用のモジュールをインポート
from filter import process_frame, id_manager
from heatmap import HeatmapGenerator
from logger import DataLogger
import reid  # 🌟【重要】特徴量を抽出するために追加

app = FastAPI()

latest_raw_frame = None
trigger_register = False  # 🌟 Webボタンが押されたかを判定するフラグ

# 🌟 NEW: UIトグル用のグローバル状態
show_heatmap_state = True
show_metrics_state = False

# --- 1. ブラウザに表示するWeb画面（HTML） ---
# --- 1. ブラウザに表示するWeb画面（HTML） ---
html_page = """
<!DOCTYPE html>
<html>
    <head>
        <title>AI監視ダッシュボード</title>
        <style>
            /* 画面全体をFlexboxにして高さを100%確保、スクロールバーを隠す */
            body { 
                font-family: sans-serif; text-align: center; background-color: #222; color: white; 
                margin: 0; padding: 20px; box-sizing: border-box;
                height: 100vh; display: flex; flex-direction: column; overflow: hidden;
            }
            h2 { color: #00ffcc; margin-top: 0; flex-shrink: 0; }
            .button-group { display: flex; justify-content: center; gap: 15px; margin-bottom: 10px; flex-shrink: 0; flex-wrap: wrap; }
            
            /* 🌟 追加：スタッフ登録用の管理パネル（初期状態は非表示） */
            #admin-panel { display: none; margin-bottom: 15px; padding: 10px; background-color: rgba(255,255,255,0.1); border-radius: 8px; }

            .btn { padding: 12px 24px; font-size: 16px; font-weight: bold; color: white; border: none; border-radius: 8px; cursor: pointer; transition: 0.2s; }
            .btn-blue { background-color: #007bff; }
            .btn-blue:hover { background-color: #0056b3; transform: scale(1.05); }
            .btn-orange { background-color: #ff9800; }
            .btn-orange:hover { background-color: #e68a00; transform: scale(1.05); }
            
            /* 🌟 追加：管理モード切り替えボタン用の地味なデザイン */
            .btn-gray { background-color: #555; padding: 12px 16px; }
            .btn-gray:hover { background-color: #777; }

            #status-msg { color: #00ffcc; font-weight: bold; height: 24px; margin-bottom: 10px; flex-shrink: 0; }

            /* 🌟 修正：カメラ映像を限界まで拡大・縮小し、比率を維持するCSS */
            .video-container {
                flex-grow: 1; 
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 0; 
                min-width: 0;
                width: 100%; /* コンテナも横幅いっぱいまで広げる */
            }
            img {
                width: 100%;  /* 🌟 追加：元のサイズを超えて限界まで拡大させる */
                height: 100%; /* 🌟 追加：元のサイズを超えて限界まで拡大させる */
                object-fit: contain; /* 縦横比を完全に維持 */
                border-radius: 10px;
                /* 💡 width:100%にしたことで枠線（border）が画面端まで広がってしまうため、
                   不要な枠線を消して背景に溶け込ませるモダンなスタイルに変更しました */
            }
        </style>
    </head>
    <body>
        <h2>🔴 リアルタイム監視ダッシュボード</h2>
        
        <div class="button-group">
            <button class="btn btn-orange" onclick="toggleHeatmap()">🔥 ヒートマップ切替 (h)</button>
            <button class="btn btn-orange" onclick="toggleMetrics()">📊 デバッグ表示切替 (i)</button>
            <button class="btn btn-gray" onclick="toggleAdminPanel()">⚙️ 管理モード</button>
        </div>

        <div class="button-group" id="admin-panel">
            <button class="btn btn-blue" onclick="registerStaff()">📸 スタッフを登録する (s)</button>
        </div>
        
        <div id="status-msg"></div>
        
        <div class="video-container">
            <img src="/video_feed" />
        </div>

        <script>
            let adminMode = false; // 管理モードの状態管理

            function showMessage(msg) {
                const el = document.getElementById('status-msg');
                el.innerText = msg;
                setTimeout(() => { el.innerText = ''; }, 3000);
            }

            // 🌟 追加：管理パネルの表示/非表示を切り替える関数
            function toggleAdminPanel() {
                adminMode = !adminMode;
                document.getElementById('admin-panel').style.display = adminMode ? 'flex' : 'none';
                showMessage(adminMode ? '🔓 管理モードを有効にしました' : '🔒 管理モードを無効にしました');
            }

            function registerStaff() {
                fetch('/api/register', { method: 'POST' })
                .then(response => response.json())
                .then(data => { showMessage(data.message); })
                .catch(error => { showMessage('⚠️ 通信エラーが発生しました'); });
            }

            function toggleHeatmap() {
                fetch('/api/toggle_heatmap', { method: 'POST' })
                .then(response => response.json())
                .then(data => { showMessage(data.message); })
                .catch(error => { showMessage('⚠️ 通信エラーが発生しました'); });
            }

            function toggleMetrics() {
                fetch('/api/toggle_metrics', { method: 'POST' })
                .then(response => response.json())
                .then(data => { showMessage(data.message); })
                .catch(error => { showMessage('⚠️ 通信エラーが発生しました'); });
            }

            document.addEventListener('keydown', function(event) {
                // 🌟 修正：'s' キーの登録は、管理モードがONの時だけ反応するようにガード
                if ((event.key === 's' || event.key === 'S') && adminMode) {
                    registerStaff();
                }
                
                if (event.key === 'h' || event.key === 'H') toggleHeatmap();
                if (event.key === 'i' || event.key === 'I') toggleMetrics();
            });
        </script>
    </body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(content=html_page)

@app.post("/api/register")
async def api_register():
    global trigger_register
    trigger_register = True
    return JSONResponse(content={"message": "✅ 登録処理を受け付けました。"})

@app.post("/api/toggle_heatmap")
async def api_toggle_heatmap():
    global show_heatmap_state
    show_heatmap_state = not show_heatmap_state
    state_str = "ON" if show_heatmap_state else "OFF"
    return JSONResponse(content={"message": f"🔥 ヒートマップ表示を {state_str} にしました。"})

@app.post("/api/toggle_metrics")
async def api_toggle_metrics():
    global show_metrics_state
    show_metrics_state = not show_metrics_state
    state_str = "ON" if show_metrics_state else "OFF"
    return JSONResponse(content={"message": f"📊 デバッグ表示を {state_str} にしました。"})

@app.websocket("/ws/upload")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global latest_raw_frame
    print("✅ カメラからの専用回線（WebSocket）が繋がりました！")
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                latest_raw_frame = img
    except WebSocketDisconnect:
        print("🔌 カメラとの通信が切断されました。")

# --- 2. 映像とAI処理を回し続けるエンジン ---
async def generate_frames(request: Request):
    global latest_raw_frame, trigger_register
    global show_heatmap_state, show_metrics_state
    
    model = YOLO("yolo26n.pt")
    heatmap_generator = HeatmapGenerator()
    data_logger = DataLogger()

    prev_display_time = time.perf_counter()
    target_frame_ms = 33.3  # 約30FPSを想定

    try:
        while True:
            if await request.is_disconnected():
                print("🔌 ブラウザとの通信が切断されました。")
                break

            if latest_raw_frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_to_process = latest_raw_frame.copy()
            latest_raw_frame = None 

            frame_start = time.perf_counter()
            capture_time = time.perf_counter()

            if trigger_register:
                trigger_register = False
                print("\n📸 Webからスタッフ登録ボタンが押されました！")
                
                max_area = 0
                best_crop = None
                
                staff_results = model(frame_to_process, verbose=False)
                if staff_results[0].boxes is not None:
                    for box, cls in zip(staff_results[0].boxes.xyxy.cpu().numpy(), staff_results[0].boxes.cls.cpu().numpy()):
                        if int(cls) == 0:  
                            x0, y0, x1, y1 = map(int, box)
                            area = (x1 - x0) * (y1 - y0)
                            if area > max_area:
                                max_area = area
                                best_crop = frame_to_process[y0:y1, x0:x1]
                
                if best_crop is not None and best_crop.shape[0] > 0 and best_crop.shape[1] > 10:
                    new_staff_feat = reid.get_feature(best_crop)
                    if hasattr(id_manager, 'staff_features'):
                        staff_dict = id_manager.staff_features
                    else:
                        staff_dict = id_manager.staff_featrues
                        
                    new_staff_id = f"S{len(staff_dict) + 1:03d}"
                    staff_dict[new_staff_id] = [new_staff_feat]
                    id_manager.save_features()
                    print(f"✅ 【登録完了】Webからスタッフ {new_staff_id} を登録しました！")
                    
                    cv2.rectangle(frame_to_process, (0, 0), (frame_to_process.shape[1], frame_to_process.shape[0]), (0, 255, 0), -1)
                else:
                    print("⚠️ 人が映っていないか、小さすぎて登録できませんでした。")

            annotated_frame, results = process_frame(
                model, frame_to_process, heatmap_generator, data_logger, 
                conf_threshold=0.6, show_heatmap=show_heatmap_state
            )

            display_time = time.perf_counter()
            processing_ms = (display_time - capture_time) * 1000.0
            frame_ms = (display_time - frame_start) * 1000.0
            time_diff = display_time - prev_display_time
            actual_fps = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_display_time = display_time
            lag_ms = max(0.0, processing_ms - target_frame_ms)

            if show_metrics_state:
                y = 30
                cv2.putText(annotated_frame, f"Persons: {len(results)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                y += 40
                cv2.putText(annotated_frame, f"Process: {processing_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y += 40
                cv2.putText(annotated_frame, f"Loop: {frame_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                y += 40
                cv2.putText(annotated_frame, f"FPS: {actual_fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y += 40
                cv2.putText(annotated_frame, f"Lag vs camera: {lag_ms:.1f} ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.001) 
            
    except asyncio.CancelledError:
        print("🛑 サーバーの終了命令を受け取りました。")
    finally:
        print("💾 リソースを解放して終了します...")
        id_manager.save_features()

@app.get("/video_feed")
async def video_feed(request: Request):
    return StreamingResponse(generate_frames(request), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("🚀 Webサーバーを起動します...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
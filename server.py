import cv2
import uvicorn
import asyncio
import numpy as np
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

# --- 1. ブラウザに表示するWeb画面（HTML） ---
html_page = """
<!DOCTYPE html>
<html>
    <head>
        <title>AI監視ダッシュボード</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #222; color: white; margin: 0; padding: 20px; }
            h2 { color: #00ffcc; }
            .btn { padding: 15px 30px; font-size: 18px; font-weight: bold; background-color: #007bff; color: white; border: none; border-radius: 8px; cursor: pointer; margin-bottom: 20px; transition: 0.2s; }
            .btn:hover { background-color: #0056b3; transform: scale(1.05); }
            .btn:active { background-color: #004085; }
            img { max-width: 90%; border: 3px solid #555; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
        </style>
    </head>
    <body>
        <h2>🔴 リアルタイム監視ダッシュボード</h2>
        
        <button class="btn" onclick="registerStaff()">📸 スタッフ登録 (画面で一番大きい人)</button>
        <br>
        
        <img src="/video_feed" />

        <script>
            function registerStaff() {
                fetch('/api/register', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    // 登録成功のメッセージをブラウザにポップアップ表示
                    alert(data.message);
                })
                .catch(error => {
                    alert('通信エラーが発生しました');
                });
            }
        </script>
    </body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(content=html_page)

# --- 🌟 NEW: Webボタンからの登録指示を受け取るAPI ---
@app.post("/api/register")
async def api_register():
    global trigger_register
    trigger_register = True  # フラグをONにする
    return JSONResponse(content={"message": "登録処理を受け付けました。画面の枠が青色になれば成功です！"})

# --- 映像受信 (WebSocket) ---
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
    
    model = YOLO("yolo26n.pt")
    heatmap_generator = HeatmapGenerator()
    data_logger = DataLogger()

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

            # 🌟【復活】Webからスタッフ登録ボタンが押された時の処理
            if trigger_register:
                trigger_register = False # フラグを戻す
                print("\n📸 Webからスタッフ登録ボタンが押されました！")
                
                max_area = 0
                best_crop = None
                
                # YOLOで推論して最大の人物を探す
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
                else:
                    print("⚠️ 人が映っていないか、小さすぎて登録できませんでした。")

            # 通常のAI処理（トラッキングと描画）
            annotated_frame, _ = process_frame(
                model, frame_to_process, heatmap_generator, data_logger, 
                conf_threshold=0.6, show_heatmap=False
            )

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
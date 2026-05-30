import cv2
import uvicorn
import asyncio
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO

from filter import process_frame, id_manager
from heatmap import HeatmapGenerator
from logger import DataLogger

app = FastAPI()

latest_raw_frame = None

html_page = """
<!DOCTYPE html>
<html>
    <head>
        <title>AI監視ダッシュボード</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #222; color: white; margin: 0; padding: 20px; }
            h2 { color: #00ffcc; }
            img { max-width: 90%; border: 3px solid #555; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
        </style>
    </head>
    <body>
        <h2>🔴 リアルタイム監視ダッシュボード</h2>
        <p>道場サーバー移行に向けた通信テスト中</p>
        <img src="/video_feed" />
    </body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(content=html_page)

# 🌟 NEW: HTTP POSTではなく、WebSocketで純粋なデータだけを超高速受信
@app.websocket("/ws/upload")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global latest_raw_frame
    print("✅ カメラからの専用回線（WebSocket）が繋がりました！")
    try:
        while True:
            # 余計な解析（ハガキの開封作業）なしで直接画像を受け取る
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                latest_raw_frame = img
    except WebSocketDisconnect:
        print("🔌 カメラとの通信が切断されました。")

async def generate_frames(request: Request):
    global latest_raw_frame
    
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
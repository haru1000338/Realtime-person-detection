import cv2
import asyncio
import websockets
import time

async def main():
    # 🌟 宛先が http:// から ws:// (WebSocket) に変わります
    SERVER_URI = "ws://localhost:8000/ws/upload"
    
    VIDEO_SOURCE = 0
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("❌ エラー: カメラを開けませんでした。")
        return

    print(f"🎥 サーバー({SERVER_URI})へ映像のストリーミングを開始します...")

    while True:
        try:
            # サーバーと電話線を繋ぐ
            async with websockets.connect(SERVER_URI) as websocket:
                print("✅ サーバーと接続しました！送信スタート！")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.1)
                        continue

                    # 画像をJPEGに圧縮（画質を70にしてさらに軽量化）
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    
                    if not ret:
                        continue

                    # 純粋な画像データだけを光の速さで送信
                    await websocket.send(buffer.tobytes())
                    
                    # カクつき防止（カメラの取得速度に合わせる微小スリープ）
                    await asyncio.sleep(0.01)

        except websockets.exceptions.ConnectionClosed:
            print("⚠️ サーバーから切断されました。再接続を探ります...")
            await asyncio.sleep(2)
        except ConnectionRefusedError:
            print("⚠️ サーバーが見つかりません。サーバー側の起動を待っています...")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        # 非同期処理として実行
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 カメラ送信を終了します。")
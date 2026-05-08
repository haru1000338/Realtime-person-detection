import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self):
        self.heatmap_accumulator = None

        self.HEAT_ADD = 2.0
        self.HEAT_MAX = 100.0
        self.DECAY_RATE = 0.98
        self.RADIUS = 20

    def apply(self, img, foot_positions):
        """
        画像と現在画面にいる人の座標リストを受け取り、ヒートマップを合成して返す
        """
        h, w = img.shape[:2]

        # ヒートマップの初期化
        if self.heatmap_accumulator is None or self.heatmap_accumulator.shape != (h, w):
            self.heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

        self.heatmap_accumulator *= self.DECAY_RATE
        
        # 足の位置にガウシアンを加算
        for (foot_x, foot_y) in foot_positions:
            cv2.circle(self.heatmap_accumulator, (foot_x, foot_y), radius=self.RADIUS, color=self.HEAT_ADD, thickness=-1)

        self.heatmap_accumulator = np.clip(self.heatmap_accumulator, 0, self.HEAT_MAX)

        # ぼかし処理
        heatmap_blurred = cv2.GaussianBlur(self.heatmap_accumulator, (0, 0), sigmaX=20, sigmaY=20)

        # 正規化（0-255の範囲）
        heatmap_normalized = (heatmap_blurred / self.HEAT_MAX * 255).astype(np.uint8)

        # サーモグラフィー風のカラーマップを適用
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # 熱がある部分だけマスクを作成
        mask = heatmap_normalized > 1  # 閾値は調整可能
        mask_3d = mask[:, :, np.newaxis]  # 3チャンネル用に次元を追加

        # 元の画像とヒートマップを合成
        alpha = 0.6  # ヒートマップの透明度
        blended = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

        overlay = np.where(mask_3d, blended, img)

        return overlay
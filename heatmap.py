import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self):
        self.heatmap_accumulator = None

        self.HEAT_ADD = 1.5
        self.HEAT_MAX = 1.0
        self.DECAY_RATE = 0.998
        # self.RADIUS = 20

        self.MIN_MAX_HEAT = 10.0

    def apply(self, img, foot_positions):
        """
        画像と現在画面にいる人の座標リストを受け取り、ヒートマップを合成して返す
        """
        h, w = img.shape[:2]
        radius = int(min(h, w) * 0.02)  # 画面サイズに応じた半径
        sigma = radius * 1.5


        # ヒートマップの初期化
        if self.heatmap_accumulator is None or self.heatmap_accumulator.shape != (h, w):
            self.heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

        self.heatmap_accumulator *= self.DECAY_RATE
        
        # 足の位置にガウシアンを加算
        for (foot_x, foot_y) in foot_positions:
            cv2.circle(self.heatmap_accumulator, (foot_x, foot_y), radius=radius, color=self.HEAT_ADD, thickness=-1)

        # ぼかし処理
        heatmap_blurred = cv2.GaussianBlur(self.heatmap_accumulator, (0, 0), sigmaX=sigma, sigmaY=sigma)

        current_max = np.max(heatmap_blurred)
        scale_max = max(current_max, self.MIN_MAX_HEAT)

        heatmap_log = np.log(heatmap_blurred + 1e-8)  # ゼロ除算を避けるための小さな値を加算
        scale_max_log = np.log(scale_max + 1e-8)
        heatmap_normalized = np.clip((heatmap_log / scale_max_log) * 255, 0, 255).astype(np.uint8)

        # サーモグラフィー風のカラーマップを適用
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # 熱がある部分だけマスクを作成
        mask = heatmap_normalized > 2  # 閾値は調整可能
        mask_3d = mask[:, :, np.newaxis]  # 3チャンネル用に次元を追加

        # 元の画像とヒートマップを合成
        alpha = 0.6  # ヒートマップの透明度
        blended = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

        overlay = np.where(mask_3d, blended, img)

        return overlay
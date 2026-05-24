import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self):
        self.heatmap_accumulator = None
        # `heatmap_accumulator`: フレームごとの熱量を蓄積するfloat32配列（HxW）。
        # 最初の呼び出し時に画像サイズで初期化され、各フレームで減衰と加算が行われる。

        # `HEAT_ADD`: 検出ごとにヒートマップに加算する値（大きいほど強く残る）。
        self.HEAT_ADD = 30
        # `HEAT_MAX`: 個々の加算やクリップに使う最大値（現在は明示的なクリップには未使用）。
        self.HEAT_MAX = 1.0
        # `DECAY_RATE`: フレームごとに累積値に掛ける減衰係数（1.0に近いほど遅くフェードする）。
        self.DECAY_RATE = 0.998
        # `RADIUS_FACTOR`: 円の半径を画像サイズに対する比率で指定する係数。
        # 例: 0.005 は min(h,w) * 0.005 ピクセルが円半径になる（最小1pxを保証）。
        self.RADIUS_FACTOR = 0.005

        # `MIN_MAX_HEAT`: 正規化の際に分母が小さくなりすぎないようにする最小値（安定化用）。
        self.MIN_MAX_HEAT = 10.0

    def apply(self, img, foot_positions, show=True):
        """
        画像と現在画面にいる人の座標リストを受け取り、ヒートマップを合成して返す
        """
        h, w = img.shape[:2]
        radius = max(1, int(min(h, w) * self.RADIUS_FACTOR))  # 画面サイズに応じた半径（最小1px）
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

        heatmap_normalized = np.clip((heatmap_blurred / scale_max) * 255, 0, 255).astype(np.uint8)

        # サーモグラフィー風のカラーマップを適用
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # 熱がある部分だけマスクを作成
        mask = heatmap_normalized > 5  # 閾値は調整可能
        mask_3d = mask[:, :, np.newaxis]  # 3チャンネル用に次元を追加

        # 元の画像とヒートマップを合成（表示が有効な場合のみ）
        if show:
            alpha = 0.6  # ヒートマップの透明度
            blended = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
            overlay = np.where(mask_3d, blended, img)
            return overlay
        else:
            # 表示しない場合でも内部累積は更新済みなので、元画像をそのまま返す
            return img
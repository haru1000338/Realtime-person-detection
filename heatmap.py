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

        # `SCALE`: ヒートマップを計算する解像度（元画像に対する倍率）。
        # ぼかし・対数・カラーマップは画素数に比例して重いので、縮小して計算し、最後に拡大して重ねる。
        # 1.0 にすると従来と同じ（元の解像度で計算）。
        self.SCALE = 0.25

    def apply(self, img, foot_positions, show=True):
        """
        画像と現在画面にいる人の座標リストを受け取り、ヒートマップを合成して返す
        """
        h, w = img.shape[:2]
        # 縮小した解像度で計算する（hs, ws は計算用のサイズ）
        hs, ws = max(1, int(h * self.SCALE)), max(1, int(w * self.SCALE))
        radius = max(1, int(min(hs, ws) * self.RADIUS_FACTOR))  # 画面サイズに応じた半径（最小1px）
        # 半径が 1px に切り上げられても、ぼかしの広がりは元の解像度のときと同じにする
        sigma = max(1.0, min(h, w) * self.RADIUS_FACTOR * 1.5 * self.SCALE)

        # ヒートマップの初期化
        if self.heatmap_accumulator is None or self.heatmap_accumulator.shape != (hs, ws):
            self.heatmap_accumulator = np.zeros((hs, ws), dtype=np.float32)

        self.heatmap_accumulator *= self.DECAY_RATE

        # 足の位置にガウシアンを加算
        for (foot_x, foot_y) in foot_positions:
            center = (int(foot_x * self.SCALE), int(foot_y * self.SCALE))
            cv2.circle(self.heatmap_accumulator, center, radius=radius, color=self.HEAT_ADD, thickness=-1)

        # 表示しない場合は、累積の更新だけで終える（ぼかし以降の重い処理を省く）
        if not show:
            return img

        # ぼかし処理
        heatmap_blurred = cv2.GaussianBlur(self.heatmap_accumulator, (0, 0), sigmaX=sigma, sigmaY=sigma)

        current_max = np.max(heatmap_blurred)
        scale_max = max(current_max, self.MIN_MAX_HEAT)

        heatmap_log = np.log(heatmap_blurred + 1e-8)  # ゼロ除算を避けるための小さな値を加算
        scale_max_log = np.log(scale_max + 1e-8)
        heatmap_normalized = np.clip((heatmap_log / scale_max_log) * 255, 0, 255).astype(np.uint8)

        # サーモグラフィー風のカラーマップを適用
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # 熱がある部分だけマスクを作成（閾値は調整可能）
        mask = (heatmap_normalized > 2).astype(np.uint8)
        if not mask.any():
            return img

        # 熱がある範囲を囲む長方形だけを元のサイズに拡大して合成する（画面全体を処理しないため速い）
        x, y, bw, bh = cv2.boundingRect(mask)
        x0, y0 = int(x / self.SCALE), int(y / self.SCALE)
        x1, y1 = min(w, int((x + bw) / self.SCALE)), min(h, int((y + bh) / self.SCALE))
        colored_roi = cv2.resize(heatmap_colored[y:y + bh, x:x + bw], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        mask_roi = cv2.resize(mask[y:y + bh, x:x + bw], (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)

        # 元の画像とヒートマップを合成
        alpha = 0.6  # ヒートマップの透明度
        overlay = img.copy()
        img_roi = overlay[y0:y1, x0:x1]
        blended = cv2.addWeighted(img_roi, 1 - alpha, colored_roi, alpha, 0)
        cv2.copyTo(blended, mask_roi, img_roi)  # mask が 1 の画素だけ blended で上書き
        return overlay

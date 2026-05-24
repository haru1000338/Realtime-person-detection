import cv2
from torchreid.utils import FeatureExtractor

# 1. モデルの準備 (OSNet)
extractor = FeatureExtractor(model_name='osnet_x1_0')
test_image_path = ''  # 画像のパスを指定

# 2. 画像の読み込みと色変換 (BGR -> RGB)
img_rgb = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)

# 3. 特徴量の抽出
features = extractor([img_rgb])

# 4. 結果の出力
print(features.shape)
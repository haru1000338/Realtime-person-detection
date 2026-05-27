import cv2
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor


def load_rgb(path):
	img = cv2.imread(path)
	if img is None:
		raise FileNotFoundError(f"Image not found: {path}")
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 1. デバイス自動検出
# 環境にCUDA対応のPyTorchがインストールされていればGPUを使う
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. モデルの準備 (OSNet)
extractor = FeatureExtractor(model_name='osnet_x1_0', device=device)

test_image_path = ''  # 画像のパスを指定
test_some_imgage_path = ''  # 画像のパスを指定
test_another_image_path = ''  # 画像のパスを指定

# 3. 画像の読み込みと色変換 (BGR -> RGB)
img_rgb = load_rgb(test_image_path)
some_img_rgb = load_rgb(test_some_imgage_path)
another_img_rgb = load_rgb(test_another_image_path)


# 4. 特徴量の抽出
features = extractor([img_rgb, some_img_rgb, another_img_rgb])

# 5. 結果の出力
print(features[0])  # 最初の画像の特徴量
print(features[1])  # 2番目の画像の特徴量
print(features[2])  # 3番目の画像の特徴量

# 6. 類似度の計算 (コサイン類似度)
similarity_1_2 = F.cosine_similarity(features[0], features[1], dim=0)
similarity_1_3 = F.cosine_similarity(features[0], features[2], dim=0)
print(f"Similarity between img_sumi1 and img_sumi2: {similarity_1_2.item():.4f}")
print(f"Similarity between img_sumi1 and img_ten1: {similarity_1_3.item():.4f}")

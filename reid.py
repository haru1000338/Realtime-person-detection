import cv2
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor

# モデルの準備 (OSNet)
# cudaが使える場合はGPUを使用し、使えない場合はCPUを使用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# torchreidをimportして利用
# OSNetのモデルがreid用のものじゃない
extractor = FeatureExtractor(model_name='osnet_x1_0', device=device)

def get_feature(img_bgr):
    """画像を受け取り、512次元の特徴量ベクトルだけを返す関数"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # リストに入れてtorchreidに渡し、0番目（1枚目）の結果を取り出す
    features = extractor([img_rgb])
    return features[0]

def compare_features(feat1, feat2):
    """抽出済みの特徴量ベクトル2つを受け取り、コサイン類似度を返す"""
    similarity = F.cosine_similarity(feat1, feat2, dim=0).item()
    return similarity
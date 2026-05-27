import cv2
import torch
import torch.nn.functional as F
from torchreid.utils import FeatureExtractor

# モデルの準備 (OSNet)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
extractor = FeatureExtractor(model_name='osnet_x1_0', device=device)

def get_similarity(test_bgr, target_bgr):

    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    features = extractor([test_rgb, target_rgb])
    test_feature, target_feature = features[0], features[1]

    # コサイン類似度を計算
    similarity = F.cosine_similarity(test_feature, target_feature, dim=0).item()
    return similarity


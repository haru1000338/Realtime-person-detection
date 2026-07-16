"""
特徴量を画像空間に逆投影してビジュアライズするデコーダ
OSNetエンコーダの逆構造を実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class ConvTransposeLayer(nn.Module):
    """Transposed convolution + bn + relu"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ConvTransposeLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FeatureDecoder(nn.Module):
    """
    512次元特徴量 → 画像空間への逆投影デコーダ
    元の入力解像度(256x128)に戻すことを目標とする
    """
    def __init__(self, feature_dim=512, height=256, width=128):
        super(FeatureDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.target_height = height
        self.target_width = width
        
        # 特徴量(512) → 512×1×1に展開するFC層逆
        self.fc_expand = nn.Sequential(
            nn.Linear(feature_dim, 512 * 8 * 4),
            nn.BatchNorm1d(512 * 8 * 4),
            nn.ReLU(inplace=True)
        )
        
        # デコーディングスタック
        # 512×8×4 → 384×16×8 (conv4逆)
        self.up_conv4 = nn.Sequential(
            ConvTransposeLayer(512, 384, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransposeLayer(384, 384, kernel_size=3, stride=1, padding=1),
        )
        
        # 384×16×8 → 256×32×16 (conv3逆)
        self.up_conv3 = nn.Sequential(
            ConvTransposeLayer(384, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransposeLayer(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # 256×32×16 → 64×64×32 (conv2逆)
        self.up_conv2 = nn.Sequential(
            ConvTransposeLayer(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransposeLayer(128, 64, kernel_size=3, stride=1, padding=1),
        )
        
        # 64×64×32 → 3×256×128 (conv1逆 + maxpool逆)
        self.up_conv1 = nn.Sequential(
            ConvTransposeLayer(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTransposeLayer(32, 3, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, features):
        """
        Args:
            features: (batch_size, 512) テンソル
        Returns:
            reconstructed: (batch_size, 3, 256, 128) 画像テンソル
        """
        # FC層逆: (batch, 512) → (batch, 512*8*4)
        x = self.fc_expand(features)
        
        # reshape: (batch, 512*8*4) → (batch, 512, 8, 4)
        x = x.view(-1, 512, 8, 4)
        
        # デコード
        x = self.up_conv4(x)  # → (batch, 384, 16, 8)
        x = self.up_conv3(x)  # → (batch, 256, 32, 16)
        x = self.up_conv2(x)  # → (batch, 64, 64, 32)
        x = self.up_conv1(x)  # → (batch, 3, 256, 128)
        
        # 正規化: [-1, 1] → [0, 255]
        x = torch.tanh(x)  # (-1, 1) に正規化
        x = (x + 1) / 2 * 255  # (0, 255) に変換
        
        return x


def tensor_to_cv2(img_tensor):
    """PyTorchテンソル → OpenCV画像に変換"""
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0]  # バッチ取得
    
    img_np = img_tensor.cpu().detach().numpy().astype(np.uint8)
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW → HWC
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np


def decode_feature(feature_tensor, decoder, device='cpu'):
    """
    単一の特徴量ベクトルをデコード
    Args:
        feature_tensor: (512,) 特徴量
        decoder: FeatureDecoder インスタンス
        device: 'cpu' or 'cuda'
    Returns:
        img_cv2: (256, 128, 3) OpenCV画像
    """
    decoder.eval()
    decoder.to(device)
    
    with torch.no_grad():
        if len(feature_tensor.shape) == 1:
            feature_tensor = feature_tensor.unsqueeze(0)  # (512,) → (1, 512)
        
        feature_tensor = feature_tensor.to(device)
        reconstructed = decoder(feature_tensor)
        img_cv2 = tensor_to_cv2(reconstructed)
    
    return img_cv2


def save_decoded_features(feature_dict, decoder_path='feature_decoder.pth', output_dir='decoded_features'):
    """
    staff_features.pt または visitor_features.pt から特徴量を読み込んで画像化して保存
    
    Args:
        feature_dict: {"S001": [feat1, feat2, ...], ...} の形式
        decoder_path: デコーダモデルの保存パス
        output_dir: 出力画像ディレクトリ
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder = FeatureDecoder().to(device)
    
    # デコーダが事前学習済みなら読み込む
    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        print(f"✅ デコーダモデルを読み込みました: {decoder_path}")
    else:
        print(f"⚠️ デコーダモデルが見つかりません（ランダム初期化で処理）: {decoder_path}")
    
    decoder.eval()
    
    for person_id, feature_list in feature_dict.items():
        person_dir = os.path.join(output_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)
        
        for idx, feature in enumerate(feature_list):
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature)
            
            img_cv2 = decode_feature(feature, decoder, device)
            
            output_path = os.path.join(person_dir, f'{person_id}_view_{idx}.png')
            cv2.imwrite(output_path, img_cv2)
            print(f"💾 保存: {output_path}")


if __name__ == "__main__":
    # テスト用
    import os
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # デコーダをテスト
    decoder = FeatureDecoder().to(device)
    dummy_features = torch.randn(4, 512).to(device)
    
    reconstructed = decoder(dummy_features)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {reconstructed.shape}")
    
    # スタッフ特徴量をデコード
    if os.path.exists('staff_features.pt'):
        staff_dict = torch.load('staff_features.pt', map_location=device)
        print(f"\n🔍 スタッフ特徴量の数: {len(staff_dict)}")
        save_decoded_features(staff_dict, output_dir='decoded_staff')
    
    # 来場者特徴量をデコード
    if os.path.exists('visitor_features.pt'):
        visitor_data = torch.load('visitor_features.pt', map_location=device)
        active_visitors = visitor_data.get('active_visitors', {})
        print(f"\n🔍 来場者特徴量の数: {len(active_visitors)}")
        save_decoded_features(active_visitors, output_dir='decoded_visitors')

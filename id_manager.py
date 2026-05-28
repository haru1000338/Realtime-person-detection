from dataclasses import dataclass

import reid


@dataclass
class IDMatchResult:
    real_id: str
    status: str
    label: str


class IDManager:
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.track_to_real_id = {}
        self.real_id_features = {}
        self.next_real_id = 1

    def _generate_real_id(self):
        real_id = f"R{self.next_real_id:03d}"
        self.next_real_id += 1
        return real_id

    def resolve(self, track_id, crop_img):
        if track_id in self.track_to_real_id:
            real_id = self.track_to_real_id[track_id]
            return IDMatchResult(real_id=real_id, status="known", label=f"ID:{track_id} Real:{real_id}")

        feature = None
        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
            feature = reid.get_feature(crop_img)

        matched_real_id = None
        best_score = 0.0

        if feature is not None:
            for real_id, past_feature in self.real_id_features.items():
                if past_feature is None:
                    continue
                score = reid.compare_features(feature, past_feature)
                if score > best_score:
                    best_score = score
                    matched_real_id = real_id

# --- ⭕ 修正後（EMAブレンドの実装） ---
        ALPHA = 0.9  # 過去の記憶をどれくらい信じるか
        
        if matched_real_id is not None and best_score >= self.similarity_threshold:
            real_id = matched_real_id
            status = f"matched:{best_score:.2f}"
            
            # 🌟 追加：姿勢の変化に追従するため、既存の特徴量と新しい特徴量をブレンドする
            if feature is not None:
                past_feat = self.real_id_features[real_id]
                self.real_id_features[real_id] = (past_feat * ALPHA) + (feature * (1.0 - ALPHA))
                
        else:
            real_id = self._generate_real_id()
            status = "new"
            # 新規人物の場合はそのまま登録
            if feature is not None:
                self.real_id_features[real_id] = feature

        self.track_to_real_id[track_id] = real_id
        return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")
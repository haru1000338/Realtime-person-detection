from dataclasses import dataclass
import os
import time
import torch
import torch.nn.functional as F

import reid

@dataclass
class IDMatchResult:
    real_id: str
    status: str
    label: str

class IDManager:
    def __init__(self, similarity_threshold=0.70, timeout_seconds=1800, visitor_path="visitor_features.pt", staff_path="staff_features.pt"):
        # 🎯 判定のライン（これを超えたら同一人物）
        self.similarity_threshold = similarity_threshold
        
        # 🎯 多様性フィルター（これ以上似ていたら辞書に追加しない）
        self.diversity_threshold = 0.90 
        
        # 🎯 1人あたりが保持する特徴量の最大数（プールサイズ）
        self.MAX_POOL_SIZE = 5 
        
        # 🎯 カメラに入ってから特徴量計算を待つフレーム数（エッジ見切れ回避）
        self.WAIT_FRAMES = 5 

        self.timeout_seconds = timeout_seconds
        self.visitor_path = visitor_path
        self.staff_path = staff_path

        # 辞書の中身は Tensor ではなく List[Tensor] になります
        self.staff_features = {}   # { "S001": [feat1, feat2, ...] }
        self.active_visitors = {}  # { "R001": [feat1, feat2, ...] }
        self.archived_visitors = {}
        
        self.track_to_real_id = {}
        self.last_seen = {}
        self.track_age = {}
        self.next_real_id = 1
    
        self.load_features()

    def _generate_real_id(self):
        real_id = f"R{self.next_real_id:03d}"
        self.next_real_id += 1
        return real_id
    
    def _clean_expired_sessions(self, current_time):
        expired_ids = []
        for real_id, last_time in self.last_seen.items():
            if real_id in self.active_visitors:
                continue
            if current_time - last_time > self.timeout_seconds:
                expired_ids.append((real_id, last_time))

        for real_id, last_time in expired_ids:
            if real_id in self.active_visitors:
                self.archived_visitors[real_id] = self.active_visitors.pop(real_id)
                print(f"🕒 ID {real_id} をアーカイブ（プール消去）しました。")
            keys_to_delete = [tid for tid, rid in self.track_to_real_id.items() if rid == real_id]
            for k in keys_to_delete:
                self.track_age.pop(k, None)
                self.track_to_real_id.pop(k, None)

    def load_features(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if os.path.exists(self.staff_path):
            try:
                loaded_staff = torch.load(self.staff_path)
                # 互換性対応: 古いデータ(Tensor単体)ならListに変換して読み込む
                for k, v in loaded_staff.items():
                    self.staff_features[k] = v if isinstance(v, list) else [v]
                print(f"👔 スタッフの特徴量を {self.staff_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ スタッフの特徴量読み込み失敗: {e}")

        if os.path.exists(self.visitor_path):
            try:
                checkpoint = torch.load(self.visitor_path, map_location=device)
                self.active_visitors = checkpoint.get("active_visitors", {})
                self.archived_visitors = checkpoint.get("archived_visitors", {})
                self.last_seen = checkpoint.get("last_seen", {})
                self.next_real_id = checkpoint.get("next_real_id", 1)
                print(f"👥 来場者の特徴量を {self.visitor_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ 来場者の特徴量読み込み失敗: {e}")

    def save_features(self):
        try:
            visitor_data = {
                "active_visitors": self.active_visitors,
                "archived_visitors": self.archived_visitors,
                "last_seen": self.last_seen,
                "next_real_id": self.next_real_id,
            }
            torch.save(visitor_data, self.visitor_path)
            torch.save(self.staff_features, self.staff_path)
        except Exception as e:
            print(f"⚠️ 特徴量ファイルの保存に失敗しました: {e}")

   # --- 🌟 NEW: プール内総当たり検索メソッド (行列演算版・最終完成版) ---
    def _find_best_match(self, query_feat, pool_dict):
        """辞書の中の「すべての姿」と総当たり戦を行い、最高スコアを返す。"""
        if query_feat is None or not pool_dict:
            return None, 0.0

        # クエリの基準となるデバイスと型を取得
        query = query_feat.detach()
        target_device = query.device
        target_dtype = query.dtype if query.is_floating_point() else torch.float32

        if query.dim() == 1:
            query = query.unsqueeze(0)

        # ⚠️ クエリをこの時点で正規化
        query = F.normalize(query, p=2, dim=1)

        all_feats = []
        id_mapping = []

        for person_id, feature_list in pool_dict.items():
            for stored_feat in feature_list:
                if stored_feat is None:
                    continue
                
                # 💡 Copilot指摘1&2: スタック前にデバイスと型をクエリに合わせる
                feat = stored_feat.to(device=target_device, dtype=target_dtype)
                
                # 💡 Copilot指摘3: 辞書内の特徴量が未正規化の場合に備えた安全策
                # （本来は保存時に正規化すべきですが、既存データとの互換性のために残します）
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                feat = F.normalize(feat, p=2, dim=1).squeeze(0)

                all_feats.append(feat)
                id_mapping.append(person_id)

        if not all_feats:
            return None, 0.0

        # デバイスが統一されているため、安全にスタック可能
        stacked_feats = torch.stack(all_feats)

        # 行列積による一括計算 (query: [1, 512], stacked_feats.T: [512, N])
        scores = torch.matmul(query, stacked_feats.T).squeeze(0)
        
        best_score_val, best_idx = torch.max(scores, dim=0)

        return id_mapping[best_idx.item()], best_score_val.item()

    def resolve(self, track_id, crop_img):
        current_time = time.time()
        self._clean_expired_sessions(current_time)

        # トラック年齢の更新と仮IDの発行
        if track_id not in self.track_age:
            self.track_age[track_id] = 1
            self.track_to_real_id[track_id] = self._generate_real_id()
        else:
            self.track_age[track_id] += 1

        real_id = self.track_to_real_id[track_id]

        # 🌟 待機期間（ロスタイム）の処理
        if self.track_age[track_id] < self.WAIT_FRAMES:
            self.last_seen[real_id] = current_time
            # すでにスタッフ昇格済みの場合は青枠を維持
            status = "staff" if real_id.startswith("S") else "waiting"
            return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")

        feature = None
        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
            raw_feature = reid.get_feature(crop_img)
            feature = torch.nn.functional.normalize(raw_feature.unsqueeze(0), p=2, dim=1).squeeze(0)  

        if feature is None:
            self.last_seen[real_id] = current_time
            status = "staff" if real_id.startswith("S") else "Unknown"
            return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")

        # ==========================================
        # 1. スタッフ辞書との照合 ＆ 動的昇格 ＆ プール更新
        # ==========================================
        matched_s_id, s_score = self._find_best_match(feature, self.staff_features)
        
        # 自身がすでにスタッフの場合の自己再評価も含む
        if matched_s_id is not None and s_score >= self.similarity_threshold:
            if not real_id.startswith("S"):
                print(f"🔄 ID昇格: トラック {track_id} が スタッフ {matched_s_id} に昇格！（スコア: {s_score:.2f}）")
                if real_id in self.active_visitors:
                    del self.active_visitors[real_id]
                real_id = matched_s_id
                self.track_to_real_id[track_id] = real_id

            # 🌟 スタッフ辞書の更新（アンカー固定 ＋ 多様性フィルター）
            if s_score < self.diversity_threshold:
                pool = self.staff_features[real_id]
                pool.append(feature) # 新しい姿を追加
                if len(pool) > self.MAX_POOL_SIZE:
                    pool.pop(1) # ⚠️ インデックス0（マスター）は絶対に消さず、1を消す！
                print(f"📈 スタッフ {real_id} の辞書が新しい姿を学習しました！(プール数: {len(pool)})")

            self.last_seen[real_id] = current_time
            return IDMatchResult(real_id=real_id, status="staff", label=f"ID:{track_id} Real:{real_id}")

        # ==========================================
        # 2. 来客辞書との照合 ＆ プール更新
        # ==========================================
        matched_v_id, v_score = self._find_best_match(feature, self.active_visitors)

        if matched_v_id is not None and v_score >= self.similarity_threshold:
            real_id = matched_v_id
            self.track_to_real_id[track_id] = real_id
            status = f"matched:{v_score:.2f}"
            
            # 🌟 来客辞書の更新（完全FIFO ＋ 多様性フィルター）
            if v_score < self.diversity_threshold:
                pool = self.active_visitors[real_id]
                pool.append(feature)
                if len(pool) > self.MAX_POOL_SIZE:
                    pool.pop(0) # ⚠️ 来客はインデックス0から容赦なく消す（完全FIFO）
        else:
            # 誰ともマッチしなかった場合、新しい来客として辞書を作成
            status = "new_visitor"
            self.active_visitors[real_id] = [feature] # 新規リストとして登録

        self.last_seen[real_id] = current_time
        return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")
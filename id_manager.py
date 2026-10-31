from dataclasses import dataclass
import os
import time
import torch

import reid

@dataclass
class IDMatchResult:
    real_id: str
    status: str
    label: str
    score: float = 0.0

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

        # 🎯 照合確定後、追跡継続中にプールへ姿を追加する間隔（フレーム数）
        self.POOL_UPDATE_INTERVAL = 15

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
        self.confirmed_tracks = {}  # { track_id: (status, score) } 照合済みのトラック
        self.next_real_id = 1
    
        self.load_features()

    def _generate_real_id(self):
        real_id = f"R{self.next_real_id:03d}"
        self.next_real_id += 1
        return real_id
    
    def _clean_expired_sessions(self, current_time):
        expired_ids = [rid for rid, last_time in self.last_seen.items()
                       if current_time - last_time > self.timeout_seconds]

        for real_id in expired_ids:
            # スタッフはアーカイブしない
            if real_id in self.active_visitors:
                self.archived_visitors[real_id] = self.active_visitors.pop(real_id)
                print(f"🕒 ID {real_id} をアーカイブ（プール消去）しました。")
            keys_to_delete = [tid for tid, rid in self.track_to_real_id.items() if rid == real_id]
            for k in keys_to_delete:
                self.track_age.pop(k, None)
                self.track_to_real_id.pop(k, None)
                self.confirmed_tracks.pop(k, None)
            del self.last_seen[real_id]

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

    def register_staff(self, feature, track_id=None):
        """スタッフを新規登録して保存し、採番したIDを返す（既存IDの最大番号+1）

        track_id を渡すと、そのトラックの来場者IDとプールをスタッフに引き継ぐ
        """
        max_num = max((int(sid[1:]) for sid in self.staff_features), default=0)
        new_staff_id = f"S{max_num + 1:03d}"
        pool = [feature]  # インデックス0 = 登録時の特徴量（アンカー）
        if track_id is not None:
            old_id = self.track_to_real_id.get(track_id)
            if old_id in self.active_visitors:
                pool += self.active_visitors.pop(old_id)[-(self.MAX_POOL_SIZE - 1):]
            self.track_to_real_id[track_id] = new_staff_id
            # 登録による確定なのでスコアは 1.0 とする
            self.confirmed_tracks[track_id] = ("staff", 1.0)
            self.last_seen[new_staff_id] = time.time()
        self.staff_features[new_staff_id] = pool
        self.save_features()
        return new_staff_id

    # --- 🌟 NEW: プール内総当たり検索メソッド ---
    def _find_best_match(self, query_feat, pool_dict):
        """辞書の中の「すべての姿」と総当たり戦を行い、最高スコアを返す"""
        best_id = None
        best_score = 0.0
        for person_id, feature_list in pool_dict.items():
            for stored_feat in feature_list:
                score = reid.compare_features(query_feat, stored_feat)
                if score > best_score:
                    best_score = score
                    best_id = person_id
        return best_id, best_score

    def _add_to_pool(self, real_id, feature, score):
        """多様性フィルターを通った特徴量をプールに追加する"""
        if score >= self.diversity_threshold:
            return
        if real_id in self.staff_features:
            pool, drop_index = self.staff_features[real_id], 1  # ⚠️ インデックス0（マスター）は絶対に消さない
        elif real_id in self.active_visitors:
            pool, drop_index = self.active_visitors[real_id], 0  # ⚠️ 来客は完全FIFO
        else:
            return
        pool.append(feature)
        if len(pool) > self.MAX_POOL_SIZE:
            pool.pop(drop_index)

    def _extract(self, crop_img):
        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
            return reid.get_feature(crop_img)
        return None

    def resolve(self, track_id, crop_img, live_track_ids=()):
        current_time = time.time()
        self._clean_expired_sessions(current_time)

        # トラック年齢の更新と仮IDの発行
        if track_id not in self.track_age:
            self.track_age[track_id] = 1
            self.track_to_real_id[track_id] = self._generate_real_id()
        else:
            self.track_age[track_id] += 1

        real_id = self.track_to_real_id[track_id]
        self.last_seen[real_id] = current_time
        label = f"ID:{track_id} Real:{real_id}"

        # 🌟 待機期間（ロスタイム）の処理
        if self.track_age[track_id] < self.WAIT_FRAMES:
            return IDMatchResult(real_id=real_id, status="waiting", label=label)

        # 🌟 照合済みのトラック：ByteTrack を信頼して照合せず、一定間隔でプールに姿を追加するだけ
        if track_id in self.confirmed_tracks:
            status, score = self.confirmed_tracks[track_id]
            if (self.track_age[track_id] - self.WAIT_FRAMES) % self.POOL_UPDATE_INTERVAL == 0:
                pool = self.staff_features.get(real_id) or self.active_visitors.get(real_id)
                feature = self._extract(crop_img)
                if pool and feature is not None:
                    own_score = max(reid.compare_features(feature, f) for f in pool)
                    # 自分のプールと似ていない姿は ID スイッチを疑って追加しない
                    if own_score >= self.similarity_threshold:
                        self._add_to_pool(real_id, feature, own_score)
            return IDMatchResult(real_id=real_id, status=status, label=label, score=score)

        # 🌟 待機明け最初の照合（特徴量が取れなければ次のフレームで再試行）
        feature = self._extract(crop_img)
        if feature is None:
            return IDMatchResult(real_id=real_id, status="Unknown", label=label)

        # 画面に映っている他のトラックの real_id は照合相手から外す
        on_screen = {self.track_to_real_id[t] for t in live_track_ids
                     if t != track_id and t in self.track_to_real_id}
        staff_pool = {k: v for k, v in self.staff_features.items() if k not in on_screen}
        visitor_pool = {k: v for k, v in self.active_visitors.items() if k not in on_screen}

        # ==========================================
        # スタッフ・来客の両方と照合し、高い方を採用（argmax）
        # ==========================================
        matched_s_id, s_score = self._find_best_match(feature, staff_pool)
        matched_v_id, v_score = self._find_best_match(feature, visitor_pool)
        best_score = max(s_score, v_score)

        if best_score >= self.similarity_threshold:
            real_id, status = (matched_s_id, "staff") if s_score >= v_score else (matched_v_id, "matched_visitor")
            self._add_to_pool(real_id, feature, best_score)
        else:
            # 誰ともマッチしなかった場合、新しい来客として辞書を作成
            status = "new_visitor"
            self.active_visitors[real_id] = [feature]

        self.track_to_real_id[track_id] = real_id
        self.confirmed_tracks[track_id] = (status, best_score)
        self.last_seen[real_id] = current_time
        return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}", score=best_score)

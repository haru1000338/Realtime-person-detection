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


class IDManager:
    def __init__(self, similarity_threshold=0.6, timeout_seconds=1800, visitor_path="visitor_features.pt", staff_path="staff_features.pt"):
        self.similarity_threshold = similarity_threshold
        self.timeout_seconds = timeout_seconds
        self.visitor_path = visitor_path
        self.staff_path = staff_path

        self.track_to_real_id = {}
        self.staff_features = {}
        self.active_visitors = {}
        self.archived_visitors = {}
        self.last_seen = {}
        
        # 🌟 NEW: トラックIDごとの生存フレーム数を記録する辞書
        self.track_age = {}
        # 🌟 NEW: 特徴量を抽出するまでに待つフレーム数（30FPSなら5フレームは約0.16秒）
        self.WAIT_FRAMES = 5

        self.next_real_id = 1
        self.TOO_SIMILAR_THRESHOLD = 0.90
        self.ALPHA = 0.9
    
        self.load_features()

    def _generate_real_id(self):
        """来場者用のユニークIDを生成"""
        real_id = f"R{self.next_real_id:03d}"
        self.next_real_id += 1
        return real_id
    
    def _clean_expired_sessions(self, current_time):
        """一定時間見かけないIDをアーカイブする"""
        expired_ids = []
        for real_id, last_time in self.last_seen.items():
            if real_id in self.active_visitors:
                continue
            if current_time - last_time > self.timeout_seconds:
                expired_ids.append((real_id, last_time))

        for real_id, last_time in expired_ids:
            if real_id in self.active_visitors:
                self.archived_visitors[real_id] = self.active_visitors.pop(real_id)
                print(f"🕒 ID {real_id} をアーカイブしました（最後の確認: {time.ctime(last_time)}）")
            # track_age のお掃除
            keys_to_delete = [tid for tid, rid in self.track_to_real_id.items() if rid == real_id]
            for k in keys_to_delete:
                self.track_age.pop(k, None)
                self.track_to_real_id.pop(k, None)

    def load_features(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if os.path.exists(self.staff_path):
            try:
                self.staff_features = torch.load(self.staff_path)
                print(f"👔 スタッフの特徴量を {self.staff_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ スタッフの特徴量ファイルの読み込みに失敗しました: {self.staff_path}")

        if os.path.exists(self.visitor_path):
            try:
                checkpoint = torch.load(self.visitor_path, map_location=device)
                self.active_visitors = checkpoint.get("active_visitors", {})
                self.archived_visitors = checkpoint.get("archived_visitors", {})
                self.last_seen = checkpoint.get("last_seen", {})
                self.next_real_id = checkpoint.get("next_real_id", 1)
                print(f"👥 来場者の特徴量を {self.visitor_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ 来場者の特徴量ファイルの読み込みに失敗しました: {self.visitor_path}")

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
            print(f"👔 スタッフの特徴量を {self.staff_path} に保存しました。")
        except Exception as e:
            print(f"⚠️ 特徴量ファイルの保存に失敗しました。")

    def resolve(self, track_id, crop_img):
        current_time = time.time()
        self._clean_expired_sessions(current_time)

        # 🌟 1. トラック年齢の更新と、仮IDの発行
        if track_id not in self.track_age:
            self.track_age[track_id] = 1
            temp_id = self._generate_real_id()
            self.track_to_real_id[track_id] = temp_id
        else:
            self.track_age[track_id] += 1

        real_id = self.track_to_real_id[track_id]

        # すでにスタッフとして昇格済みの場合は、再計算せずにそのまま返す
        if real_id.startswith("S"):
            self.last_seen[real_id] = current_time
            return IDMatchResult(real_id=real_id, status="staff", label=f"ID:{track_id} Real:{real_id}")

        # 🌟 2. 待機期間（ロスタイム）の処理
        if self.track_age[track_id] < self.WAIT_FRAMES:
            # まだ指定フレームに達していないので、重い特徴量計算（OSNet）はサボる
            self.last_seen[real_id] = current_time
            return IDMatchResult(real_id=real_id, status="waiting", label=f"ID:{track_id} Real:{real_id}")

        # 🌟 3. 待機明け：全身が映った（はずの）綺麗な画像で特徴量抽出
        feature = None
        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
            feature = reid.get_feature(crop_img)

        if feature is None:
            self.last_seen[real_id] = current_time
            return IDMatchResult(real_id=real_id, status="Unknown", label=f"ID:{track_id} Real:{real_id}")

        # 🌟 4. スタッフ認証（動的ID昇格テスト）
        matched_staff_id = None
        best_score = 0.0

        for s_id, staff_feat in self.staff_features.items():
            score = reid.compare_features(feature, staff_feat)
            if score > best_score:
                best_score = score
                matched_staff_id = s_id

        if matched_staff_id is not None and best_score >= self.similarity_threshold:
            # ！！ここで仮の来客IDを捨てて、スタッフIDに昇格させる！！
            self.track_to_real_id[track_id] = matched_staff_id
            real_id = matched_staff_id
            
            # 昇格した不要な来客IDのゴミデータがあれば消す
            if real_id in self.active_visitors:
                del self.active_visitors[real_id]
                
            print(f"🔄 ID昇格: トラック {track_id} が スタッフ {real_id} に昇格しました！（スコア: {best_score:.2f}）")
            self.last_seen[real_id] = current_time
            return IDMatchResult(real_id=real_id, status="staff", label=f"ID:{track_id} Real:{real_id}")
        
        # 🌟 5. 来客としての処理（EMAブレンド）
        matched_visitor_id = None
        best_visitor_score = 0.0
        for visitor_id, visitor_feat in self.active_visitors.items():
            if visitor_id == real_id: # 自分自身とは比較しない
                continue
            score = reid.compare_features(feature, visitor_feat)
            if score > best_visitor_score:
                best_visitor_score = score
                matched_visitor_id = visitor_id

        if matched_visitor_id is not None and best_visitor_score >= self.TOO_SIMILAR_THRESHOLD:
            # 過去の来客と完全に一致した場合、仮IDを捨てて過去のIDを引き継ぐ
            self.track_to_real_id[track_id] = matched_visitor_id
            real_id = matched_visitor_id
            self.last_seen[real_id] = current_time
            status = f"matched:{best_visitor_score:.2f}"
            
        elif matched_visitor_id is not None and best_visitor_score >= self.similarity_threshold:
            self.track_to_real_id[track_id] = matched_visitor_id
            real_id = matched_visitor_id
            past_feat = self.active_visitors[real_id]
            self.active_visitors[real_id] = ((past_feat * self.ALPHA) + (feature * (1.0 - self.ALPHA)))
            self.last_seen[real_id] = current_time
            status = f"matched:{best_visitor_score:.2f}"
            
        else:
            # 誰ともマッチしなかった場合、仮発行していたIDを名簿（active_visitors）に正式登録
            self.active_visitors[real_id] = feature
            self.last_seen[real_id] = current_time
            status = "new_visitor"

        return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")
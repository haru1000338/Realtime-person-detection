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
    def __init__(self, similarity_threshold=0.7, timeout_seconds=1800, visitor_path="visitor_features.pt", staff_path="staff_features.pt"):
        self.similarity_threshold = similarity_threshold
        self.timeout_seconds = timeout_seconds
        self.visitor_path = visitor_path
        self.staff_path = staff_path

        self.track_to_real_id = {}
        self.staff_featrues = {}
        self.active_visitors = {}
        self.archived_visitors = {}
        self.last_seen = {}

        self.next_real_id = 1
        self.TOO_SIMILAR_THRESHOLD = 0.90
        self.ALPHA = 0.9
    
        self.load_features()

    def _generate_real_id(self):
        real_id = f"R{self.next_real_id:03d}"
        self.next_real_id += 1
        return real_id
    
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

        for real_id in expired_ids:
            if real_id in self.active_visitors:
                self.archived_visitors[real_id] = self.active_visitors.pop(real_id)
                print(f"🕒 ID {real_id} をアーカイブしました（最後の確認: {time.ctime(self.last_seen[real_id])}）")

    def load_features(self):
        """スタッフと来場者の特徴量をファイルから読み込む"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # スタッフの特徴量を読み込む（存在する場合）
        if os.path.exists(self.staff_path):
            try:
                self.staff_featrues = torch.load(self.staff_path)
                print(f"👔 スタッフの特徴量を {self.staff_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ スタッフの特徴量ファイルの読み込みに失敗しました: {self.staff_path}")

        # 来場者の特徴量を読み込む（存在する場合）
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
        """終了時にスタッフと来場者の特徴量をファイルに保存する"""
        try:
            visitor_data = {
                "active_visitors": self.active_visitors,
                "archived_visitors": self.archived_visitors,
                "last_seen": self.last_seen,
                "next_real_id": self.next_real_id,
            }
            torch.save(visitor_data, self.visitor_path)
            torch.save(self.staff_featrues, self.staff_path)
            print(f"👔 スタッフの特徴量を {self.staff_path} に保存しました。")
        except Exception as e:
            print(f"⚠️ スタッフの特徴量ファイルの保存に失敗しました: {self.staff_path}")


    def resolve(self, track_id, crop_img):
        current_time = time.time()
        self._clean_expired_sessions(current_time)

        if track_id in self.track_to_real_id:
            real_id = self.track_to_real_id[track_id]
            if real_id in self.last_seen:
                self.last_seen[real_id] = current_time
            
            status = "staff" if real_id.startswith("S") else "known"
            return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")

        feature = None
        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
            feature = reid.get_feature(crop_img)

        if feature is  None:
            real_id = f"R{self.next_real_id:03d}"
            self.track_to_real_id[track_id] = real_id
            return IDMatchResult(real_id=real_id, status="Unknown", label=f"ID:{track_id} Real:{real_id}")

# --- ⭕ 修正後（EMAブレンドの実装） ---
        matched_staff_id = None
        best_score = 0.0

        for real_id, staff_feat in self.staff_featrues.items():
            score = reid.compare_featrues(feature, staff_feat)
            if score > best_score:
                best_score = score
                matched_staff_id = real_id

        ALPHA = 0.9  # 過去の記憶をどれくらい信じるか

        if matched_staff_id is not None and best_score >= self.similarity_threshold:
            real_id = matched_staff_id
            self.track_to_real_id[track_id] = real_id
            print(f"✅ トラックID {track_id} はスタッフID {real_id} とマッチしました（スコア: {best_score:.2f}）")
            return IDMatchResult(real_id=real_id, status="staff", label=f"ID:{track_id} Real:{real_id}")
        
        matched_visitor_id = None
        best_visitor_score = 0.0
        for visitor_id, visitor_feat in self.active_visitors.items():
            score = reid.compare_features(feature, visitor_feat)
            if score > best_visitor_score:
                best_visitor_score = score
                matched_visitor_id = visitor_id
        if matched_visitor_id is not None and best_visitor_score >= self.TOO_SIMILAR_THRESHOLD:
            real_id = matched_visitor_id
            status = f"matched:{best_visitor_score:.2f}"
            self.last_seen[real_id] = current_time
        elif matched_visitor_id is not None and best_visitor_score >= self.similarity_threshold:
            real_id = matched_visitor_id
            status = f"matched:{best_visitor_score:.2f}"
            past_feat = self.active_visitors[real_id]
            self.active_visitors[real_id] = ((past_feat * self.ALPHA) + (feature * (1.0 - self.ALPHA)))
            self.last_seen[real_id] = current_time
        else:
            real_id = self._generate_real_id()
            status = "new"
            self.active_visitors[real_id] = feature
            self.last_seen[real_id] = current_time

        self.track_to_real_id[track_id] = real_id
        return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")
from dataclasses import dataclass
import os
import time
import torch

import reid

# TODO: dataclass: 変数名に型を付ける？（書き換えたらどうなるのか要調査）
@dataclass
class IDMatchResult:
    real_id: str
    status: str
    label: str

class IDManager:
    def __init__(self, similarity_threshold=0.70, timeout_seconds=1800, visitor_path="visitor_features.pt", staff_path="staff_features.pt"):
        """
        similarity_threshold: コサイン類似度の閾値（これ以上似ていたら同一人物とみなす）
        diversity_threshold: 多様性フィルター（これ以上似ていたら辞書に追加しない）
        MAX_POOL_SIZE: 1人に対して保持する特徴量の最大数
        WAIT_FRAMES: カメラに入ってから特徴量計算を待つフレーム数
        timeout_seconds: 来客を比較対象から外すまでの時間（秒）
        visitor_path: 来客の特徴量を保存するファイルパス
        staff_path: スタッフの特徴量を保存するファイルパス
        staff_features: feat1は基準、feat2以降は多様性のための姿勢の追加を保持する辞書 {staff_id: [feature1, feature2, ...]}
        active_visitors: 現在アクティブな来客の特徴量を保持する辞書。上位を保持するためのFIFOリスト（登録時の特徴量が先頭） {visitor_id: [feature1, feature2, ...]}
        archived_visitors: 一定時間経過した来客の特徴量を保持する辞書 {visitor_id: [feature1, feature2, ...]}
        track_to_real_id: トラックIDと来客ID(real_id)の対応を保持する辞書 {track_id: real_id}
        last_seen: 観測されたIDと最後に観測された時刻を保持する辞書 {real_id: last_seen_time}
        track_age: track_idが何フレーム観測されているかを保持する辞書 {track_id: age_in_frames}
        next_real_id: 新しい来客IDの生成用カウンタ
        """
        self.similarity_threshold = similarity_threshold
        self.diversity_threshold = 0.90 
        self.MAX_POOL_SIZE = 5 
        self.WAIT_FRAMES = 5 
        self.timeout_seconds = timeout_seconds
        self.visitor_path = visitor_path
        self.staff_path = staff_path

        # 辞書の初期化
        self.staff_features = {}   # { "S001": [feat1, feat2, ...] }
        self.active_visitors = {}  # { "R001": [feat1, feat2, ...] }
        self.archived_visitors = {}
        self.track_to_real_id = {}
        self.last_seen = {}
        self.track_age = {}
        self.next_real_id = 1
    
        self.load_features()

    def _generate_real_id(self):
        """新たな来客IDを生成する→理想は一人一つのID(real_id)を持つこと"""
        real_id = f"R{self.next_real_id:03d}"
        self.next_real_id += 1
        return real_id
    
    def _clean_expired_sessions(self, current_time):
        """
        期限切れのセッションをクリーンアップする
        TODO: 一つ目のfor文でミスあり？（unactiveならcontinueのほうが正確？）
        TODO: 二つ目のfor文にもミス？（expired_idsの条件とactive_visitorsの条件が一致しない？）
        expired_idsの条件: 期限切れ＋active_visitorsに存在しない
        確認済み: アーカイブ処理が高確率で動いていない可能性がある→active_visitorsに存在する場合はスキップしているため
        """
        expired_ids = []
        # IDごとに確認処理
        for real_id, last_time in self.last_seen.items():
            # すでにactive_visitorsに存在する場合はスキップ
            if real_id in self.active_visitors:
                continue
            # 最後に観測されてからどれくらい経ったか（現在時間ー観測時間）→期限切れかどうかを判断
            if current_time - last_time > self.timeout_seconds:
                expired_ids.append((real_id, last_time))

        # 期限切れリストのIDを確認
        for real_id, last_time in expired_ids:
            # active_visitorsに存在する場合はアーカイブに移動し、active_visitorsから削除する
            if real_id in self.active_visitors:
                self.archived_visitors[real_id] = self.active_visitors.pop(real_id)
                print(f"🕒 ID {real_id} をアーカイブ（プール消去）しました。")

            # real_idに対応するtrack_idを削除する
            keys_to_delete = [tid for tid, rid in self.track_to_real_id.items() if rid == real_id]
            for k in keys_to_delete:
                self.track_age.pop(k, None)
                self.track_to_real_id.pop(k, None)

    def load_features(self):
        """
        特徴量をファイルから読み込む->形式を統一
        staff_features: {staff_id: [feature1, feature2, ...]}
        visitor_features: {
            active_visitors: {visitor_id: [feature1, feature2, ...]},
            archived_visitors: {...},
            last_seen: {...},
            next_real_id: int
        }
        """
        # map_locationを指定して、どこに特徴量のテンソルをロードするかを決めるために必要
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # スタッフ
        if os.path.exists(self.staff_path):
            try:
                # TODO: map_locationが必要そう
                loaded_staff = torch.load(self.staff_path)
                # 互換性対応: 古いデータ(Tensor単体)ならListに変換して読み込む
                for k, v in loaded_staff.items():
                    self.staff_features[k] = v if isinstance(v, list) else [v]
                print(f"👔 スタッフの特徴量を {self.staff_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ スタッフの特徴量読み込み失敗: {e}")

        # 来客
        if os.path.exists(self.visitor_path):
            try:
                checkpoint = torch.load(self.visitor_path, map_location=device)  # map_location: 保存されたテンソルをロードするデバイスの指定

                # staffと特徴量の保存形式が違う→ active_visitors, archived_visitors, last_seen, next_real_idを読み込む
                self.active_visitors = checkpoint.get("active_visitors", {})
                self.archived_visitors = checkpoint.get("archived_visitors", {})
                self.last_seen = checkpoint.get("last_seen", {})
                self.next_real_id = checkpoint.get("next_real_id", 1)
                print(f"👥 来場者の特徴量を {self.visitor_path} から読み込みました。")
            except Exception as e:
                print(f"⚠️ 来場者の特徴量読み込み失敗: {e}")

    def save_features(self):
        """
        来客とスタッフの特徴量をvisitor_pathかstaff_pathに保存する
        保存するタイミングを分けた方が良いかも？
        """
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

    def _find_best_match(self, query_feat, pool_dict):
        """pool_dict（staff_features or active_visitors）内のすべての特徴量とquery_featureを比較→最も近いIDとコサイン類似度を返す"""
        best_id = None
        best_score = 0.0
        for person_id, feature_list in pool_dict.items():
            for stored_feat in feature_list:
                # コサイン類似度を計算
                score = reid.compare_features(query_feat, stored_feat)
                if score > best_score:
                    best_score = score
                    best_id = person_id
        return best_id, best_score

    def resolve(self, track_id, crop_img):
        """
        Track IDと切り抜き画像を受け取り、IDを決定する
        track_idとreal_idの更新→スタッフor来客の判定→IDの更新→辞書の更新→返り値
        返り値: IDMatchResult(real_id, status, label)
        """
        current_time = time.time()
        self._clean_expired_sessions(current_time)

        # トラックIDが初めて観測された場合、track_ageを1に設定し、real_idを生成する
        # real_idの発行→置き換えのため、real_idと来場した人数は対応していない
        if track_id not in self.track_age:
            self.track_age[track_id] = 1
            self.track_to_real_id[track_id] = self._generate_real_id()
        else:
            self.track_age[track_id] += 1

        real_id = self.track_to_real_id[track_id]

        # 待機フレーム中の対応
        if self.track_age[track_id] < self.WAIT_FRAMES:
            self.last_seen[real_id] = current_time
            # この段階のステータスはstaff or waiting（すでにスタッフ昇格済みの場合は青枠を維持）
            status = "staff" if real_id.startswith("S") else "waiting"
            # TODO: 返り値は変更したほうよさそう？（IDとRealだと紛らわしい？）
            return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")

        # 待機フレーム終了後、特徴量を抽出
        feature = None   # 抽出した特徴量（512次元ベクトル）を保持する変数

        # 縦0または横10以下の画像は無視する（特徴量抽出しない）
        if crop_img is not None and crop_img.shape[0] > 0 and crop_img.shape[1] > 10:
            feature = reid.get_feature(crop_img)

        # もし特徴量が抽出できなかった場合、IDを更新せずに終了する
        if feature is None:
            self.last_seen[real_id] = current_time
            status = "staff" if real_id.startswith("S") else "Unknown"
            return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")

        # スタッフと比較した結果、一番類似度が高いスタッフIDとスコアを取得
        matched_s_id, s_score = self._find_best_match(feature, self.staff_features)

        # TODO: 処理する順番が超重要だけど、先にスタッフと比較したスコアと来客と比較したスコアを比較して、どちらが高いかで処理を分けるのが良いかも？
        # TODO: 数値判断で、閾値で本当に分かれているのなら順番だけで判断しても良いかも？（スタッフと来客の閾値を同じにする）

        # スタッフ比較時のスコアが閾値以上なら
        if matched_s_id is not None and s_score >= self.similarity_threshold:
            # スタッフ昇格処理（active_visitorsから削除->real_idを更新）
            if not real_id.startswith("S"):
                print(f"🔄 ID昇格: トラック {track_id} が スタッフ {matched_s_id} に昇格！（スコア: {s_score:.2f}）")
                if real_id in self.active_visitors:
                    del self.active_visitors[real_id]
                real_id = matched_s_id
                self.track_to_real_id[track_id] = real_id

            # スタッフ辞書の更新（多様性フィルター）
            if s_score < self.diversity_threshold:
                pool = self.staff_features[real_id]
                pool.append(feature) # 新しい姿を追加
                # FIFOで古い姿を削除（マスターは絶対に消さない）
                if len(pool) > self.MAX_POOL_SIZE:
                    pool.pop(1) # ⚠️ インデックス0（マスター）は絶対に消さず、1を消す！
                print(f"📈 スタッフ {real_id} の辞書が新しい姿を学習しました！(プール数: {len(pool)})")

            self.last_seen[real_id] = current_time
            # TODO: 返り値は変更したほうよさそう？（IDとRealだと紛らわしい？）
            return IDMatchResult(real_id=real_id, status="staff", label=f"ID:{track_id} Real:{real_id}")


        # スタッフとマッチしなかった場合、来客(active_visitors)と比較
        matched_v_id, v_score = self._find_best_match(feature, self.active_visitors)

        # 来客比較時のスコアが閾値以上なら
        if matched_v_id is not None and v_score >= self.similarity_threshold:
            real_id = matched_v_id
            self.track_to_real_id[track_id] = real_id
            status = f"matched:{v_score:.2f}"
            
            # 来客辞書の更新（多様性フィルター）
            if v_score < self.diversity_threshold:
                pool = self.active_visitors[real_id]
                pool.append(feature)
                if len(pool) > self.MAX_POOL_SIZE:
                    pool.pop(0) # 完全FIFO
        else:
            # 誰ともマッチしなかった場合、新しい来客として辞書を作成
            status = "new_visitor"
            self.active_visitors[real_id] = [feature] # 新規リストとして登録

        self.last_seen[real_id] = current_time
        return IDMatchResult(real_id=real_id, status=status, label=f"ID:{track_id} Real:{real_id}")
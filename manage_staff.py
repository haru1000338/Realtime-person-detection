import torch
import os

def main():
    file_path = "staff_features.pt"
    
    # ファイルが存在するかチェック
    if not os.path.exists(file_path):
        print("⚠️ まだスタッフデータが作成されていません。")
        return

    # デバイスを判定して安全に読み込み
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = torch.load(file_path, map_location=device)

    while True:
        print("\n" + "="*40)
        print(f" 👔 現在の登録スタッフ (計 {len(features)} 名)")
        print("="*40)
        
        if len(features) == 0:
            print("  (登録なし)")
        else:
            for sid in sorted(features.keys()):
                print(f"  - {sid}")

        print("\n操作を選んでください:")
        print("[1] 特定のIDを削除する (例: 複数回登録してしまったS002を消す)")
        print("[2] 全データをリセットする (本番前の初期化など)")
        print("[q] 終了する")
        
        choice = input(">> ").strip()

        if choice == 'q':
            print("👋 管理ツールを終了します。")
            break
            
        elif choice == '1':
            target = input("🗑️ 削除したいIDを入力してください（例: S001）: ").strip().upper()
            if target in features:
                del features[target]
                # 削除後、即座に上書き保存
                torch.save(features, file_path)
                print(f"✅ {target} の特徴量データを完全に削除しました！")
            else:
                print(f"❌ {target} は見つかりません。入力ミスがないか確認してください。")
                
        elif choice == '2':
            confirm = input("⚠️ 本当に全スタッフデータを消去しますか？ (y/n): ").strip().lower()
            if confirm == 'y':
                features.clear()
                torch.save(features, file_path)
                print("💥 全データをリセットしました！")
        else:
            print("正しく入力してください。")

if __name__ == "__main__":
    main()
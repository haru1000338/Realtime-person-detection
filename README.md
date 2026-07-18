# Track Eye
リアルタイム人物検出と滞在時間分析アプリのREADME兼メモ

## 概要
このアプリは，カメラ映像から人物をリアルタイムに検出・追跡し，特定エリア内での滞在時間を記録・可視化することを目的として開発するAIアプリである．
ユーザがカメラ映像を入力すると，AIが人物検出，追跡，再同定，滞在時間分析を行い，ダッシュボードとログとして結果を出力する．

## 想定ユーザ
* オープンキャンパスや展示会の来場者動線を分析したい人
* スタッフと来場者を区別しながら人物追跡したい人
* カメラ映像から滞在時間やヒートマップを簡単に確認したい人

## 主な機能
* カメラ映像からの人物検出・追跡機能
* OSNet を使った人物再同定（Re-ID）機能
* ブースごとの滞在時間記録・CSV出力機能
* Streamlit による分析ダッシュボード表示機能
* ヒートマップやデバッグ情報の表示切り替え機能


## 使用する技術

### フロントエンド

* HTML / CSS / JavaScript
* Streamlit

### バックエンド

* Python
* FastAPI
* WebSocket
* OpenCV

### AIモデル・API

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [ByteTrack](https://github.com/ifzhang/ByteTrack)
* [Torchreid OSNet](https://github.com/KaiyangZhou/deep-person-reid)

## 入力と出力

### 入力

* カメラ映像
* WebSocket 経由のフレームデータ
* スタッフ登録用の人物特徴量

### 出力

* 追跡結果付きの映像
* 滞在時間のCSVログ
* Streamlit の分析結果
* ヒートマップ表示

## 工夫したい点

* Re-ID により，カメラ外に出た人物もできるだけ同一人物として追跡する
* スタッフと来場者の特徴量を分けて管理し，誤認識を減らす
* 数フレーム待ってから特徴量を保存し，エッジ見切れの影響を減らす
* 滞在時間やヒートマップを見やすく表示する

## 今後やりたいこと

* [ ] 画面イメージを作る
* [ ] 表示項目を整理する
* [ ] Re-ID の精度と速度をさらに調整する
* [ ] 入力フォームや設定画面を追加する
* [ ] 出力結果をより見やすく表示する
* [ ] デザインを整える
* [ ] 発表用のデモを作る

## セットアップ方法と実行方法

### 必要なライブラリのインストール

Docker でサーバーを動かす場合は，コンテナ側で依存関係を用意するため，基本的には追加インストールは不要です．
ローカルでカメラクライアントを動かす場合は，必要に応じて以下を実行します．

```bash
pip install opencv-python websockets
```

### 実行方法

このアプリは，AIサーバー，ダッシュボード，カメラクライアントの3つを起動して使います．

1. Docker でサーバーとダッシュボードを起動する

```bash
docker compose up -d --build
docker compose exec [コンテナ名] bash
python server.py
```

2. 別ターミナルでダッシュボードを起動する

```bash
docker compose exec [コンテナ名] bash
streamlit run app.py
```

3. カメラ映像を送信するクライアントをローカルで起動する

```bash
python camera_client.py
```

`camera_client.py` の `SERVER_URI` は，実際のサーバーアドレスに合わせて変更してください．
同じPCで動かす場合は `ws://localhost:8000/ws/upload` のままで問題ありません．
また，接続するカメラが複数ある場合は `VIDEO_SOURCE` も適宜変更してください．

## ファイル構成

```text
.
├── README.md
├── app.py
├── server.py
├── camera_client.py
├── filter.py
├── id_manager.py
├── reid.py
├── visualizer.py
├── heatmap.py
├── zone_analytics.py
├── logger.py
├── requirements.txt
└── data/
```

## 参考リンク

* [Ultralytics](https://github.com/ultralytics/ultralytics)
* [ByteTrack](https://github.com/ifzhang/ByteTrack)
* [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
* [Streamlit](https://streamlit.io/)

# リアルタイム人物検出と滞在時間分析アプリ(Track Eye)

| | URL |
|---|---|
| アプリケーション | https://trackeye.fsmlabo.org/ |
| ダッシュボード | https://trackeye-dashboard.fsmlabo.org/ |

※ サーバ側のプロセスが起動している間のみアクセスできます。

## 開発目的

オープンキャンパスにおける来場者の動線分析を目的としたアプリケーションです。カメラ映像からリアルタイムに人物を検出・追跡し、特定のエリア内での滞在時間を記録・可視化することで、来場者の行動パターンを把握し、イベント運営の改善に役立てることができます。また、スタッフを登録することで、特定の人物を追跡しやすくする機能も備えています。

## システム概要

このアプリケーションは、カメラ映像からリアルタイムに人物を検出・追跡し、特定のエリア内での滞在時間を記録・可視化するシステムです。最大の特徴はAIによる人物再同定（Re-ID: OSNet）を組み込んでいる点です。単なる物体追跡（トラッキング）では、カメラの画角から外れるたびに違うIDが割り当てられるという課題があります。本システムでは、OSNetを用いて人ごとに特徴量をベクトル化して比較することで、来場者とスタッフを区別することができます。また、スタッフがカメラの画角から外れても同一人物として追跡を続けることが可能になります。

また、FastAPIを用いたWebSocket通信による映像伝送を実装することで、サーバに映像処理を集約し、クライアント側の負荷を軽減しています。これにより、処理の重いAIモデルをサーバ側で実行し、クライアントは映像の送信と簡単な操作に専念できる構成になっています。

## このアプリのアピールポイント

- Re-ID（人物再同定）による個体識別：トラッキングが途切れても同一人物を認識し、スタッフと来場者を区別できる
- スタッフ登録機能：運営スタッフの特徴量を事前に登録し、来場者とは別の辞書で管理することで、スタッフを優先的に判定する
- ノイズへの対策：カメラの画角に入る瞬間ではなく、数フレーム後の特徴量を保存することで、誤認識のリスクを減らす工夫
- FastAPI と WebSocket を用いた映像伝送：サーバ側で映像処理を集約し、クライアントの負荷を軽減する構成
- 特徴量の保存方法：同じIDに対して複数の特徴量を保存することで、人物の見た目の変化や誤認識に対する耐性を向上させる工夫
- 精度検証の仕組み：`eval_reid.py` により、本番パイプラインを通さずにOSNet単体の類似度を測定できる

## 全体のアーキテクチャと処理の流れ

1. 映像取得：`camera_client.py` がカメラ映像を取得し、WebSocketを通じてサーバに送信
2. AI処理(YOLO/ByteTrack)：`server.py` が映像を受け取り、`filter.py` がYOLOで人物を検出し、ByteTrackで一時的な追跡ID（Track_ID）を割り当てる
3. Re-ID処理(OSNet)：`id_manager.py` が追跡IDごとにOSNetで特徴量を抽出し、登録済みの特徴量と比較して同一人物かどうかを判定し、恒久的なID（Real_ID）を割り当てる
4. 空間分析：`zone_analytics.py` が足元座標をもとにポリゴン領域内にいるかどうかを判定し、滞在時間を計算する。退出時に `logger.py` がCSVに記録する
5. ヒートマップの生成と表示：`heatmap.py` が人物の移動経路を可視化し、`visualizer.py` が枠・ラベル・軌跡を描画して映像に重ねる
6. 分析結果の表示：`app.py` でStreamlitを用いて集計結果をダッシュボード形式で表示する

## 使用技術

### AIモデル

- YOLO(Ultralytics)：リアルタイム物体検出のためのモデル。人物を検出するために使用。
- ByteTrack：物体追跡のためのアルゴリズム。検出された人物に一時的な追跡IDを割り当てるために使用。設定は `bytetrack.yaml` に記述。
- OSNet(Torchreid)：人物再同定のためのモデル。追跡IDごとに特徴量を抽出し、同一人物かどうかを判定するために使用。

### バックエンド・通信

- FastAPI：WebサーバとAPIの構築に使用。WebSocket通信を実装して、クライアントから映像を受け取るために使用。
- WebSocket：クライアントとサーバ間でリアルタイムに映像を送受信するために使用。

### データ分析・ダッシュボード

- streamlit：分析結果をダッシュボード形式で表示するために使用。
- pandas：データの集計や分析に使用。
- plotly：ダッシュボード上でのグラフ描画に使用。
- OpenCV：映像処理全般、およびヒートマップの生成・合成に使用。

## ディレクトリ構成

```
Realtime-person-detection/
├── server.py                    AIサーバ（FastAPI）
├── camera_client.py             カメラクライアント
├── app.py                       ダッシュボード（Streamlit）
├── filter.py                    フレーム単位の検出・追跡・描画
├── id_manager.py                Re-IDによるID管理
├── zone_analytics.py            ブース判定と滞在時間の計算
├── logger.py                    CSVへの記録
├── heatmap.py                   ヒートマップ生成
├── visualizer.py                枠・ラベル・軌跡の描画
├── eval_reid.py                 OSNet単体の類似度検証スクリプト
│
├── Dockerfile                   実行環境の定義
├── docker-compose.example.yml   Docker環境の設定例
├── requirements.txt             依存ライブラリ
├── bytetrack.yaml               ByteTrackの設定
├── DATAFLOW.md                  データフロー設計図
│
├── docs/
│   └── experiments/             評価実験の記録と結果CSV（命名規則は同ディレクトリのREADMEを参照）
│
├── tests/                       検証用データ（gitでは追跡しない）
│   └── <実験ID>/
│       ├── input/               入力画像
│       ├── yolo-crops/          YOLOが切り出した画像
│       └── results/             結果CSV・ログ
│
└── production/                  本番（イベント）の実行履歴（gitでは追跡しない）
```

`osnet_weights/` に配置するOSNetの学習済み重み、および撮影した画像はgitで追跡していません。

## 使用方法（セットアップと実行）

本システムは、**「AIサーバー（Docker）」** と **「カメラクライアント（ローカル）」** の2つを起動することで動作します。

### 1. サーバー側の起動（Docker環境）

NVIDIA GPUを活用したAIサーバーを立ち上げます。サーバーマシン上で以下のコマンドを実行してください。

まず、`docker-compose.example.yml` を参考に `docker-compose.yml` を作成し、環境に合わせてポート番号を調整します。

```bash
# 設定ファイルの作成（初回のみ）
cp docker-compose.example.yml docker-compose.yml

# コンテナのビルドとバックグラウンド起動
docker compose up -d --build
```

次に、コンテナ内に入ってAIサーバー(Track Eye)を起動します。

```bash
# ホスト側で実行：コンテナ内のシェルに入る
docker compose exec trackeye bash
```

```bash
# コンテナ内で実行：AIサーバーの起動
python server.py
# -> 起動後、ブラウザで https://trackeye.fsmlabo.org/ にアクセス
```

別のターミナルを開き、同じ手順でダッシュボードも起動します。

```bash
# ホスト側で実行：コンテナ内のシェルに入る
docker compose exec trackeye bash
```

```bash
# コンテナ内で実行：ダッシュボードの起動
streamlit run app.py
# -> 起動後、ブラウザで https://trackeye-dashboard.fsmlabo.org/ にアクセス
```

`trackeye` は `docker-compose.yml` で定義したサービス名です。変更した場合は読み替えてください。
コンテナ内の 8000 番（`server.py`）と 8501 番（`app.py`）が、それぞれホスト側のポートに公開されます。公開ポートは `docker-compose.yml` の `ports` を参照してください。

### 2. カメラクライアント側の起動（ローカル環境）

物理カメラが接続されているPCで実行します。`server.py` が起動していることを確認したうえで、`camera_client.py` の接続先をサーバーに合わせてください。また、カメラの接続番号（`VIDEO_SOURCE`）も適宜変更してください。（例: `VIDEO_SOURCE = 0`）

```python
SERVER_URI = "ws://<サーバーのIPアドレス>:<公開ポート>/ws/upload"
```

同じPCでサーバーもカメラも動かす場合は、`ws://localhost:8000/ws/upload` のままで問題ありません。

なお、公開ドメインは Cloudflare Access による認証で保護されているため、ブラウザ以外からのアクセスには対応していません。`camera_client.py` からはサーバへ直接接続してください。

```bash
# 1. 必要なライブラリのインストール（初回のみ）
pip install opencv-python websockets

# 2. カメラ映像の送信開始
python camera_client.py
```

### 接続できない場合の確認

サーバーとクライアントが別のネットワークにある場合、直接は接続できないことがあります。接続先のポートに到達できるかを先に確認してください。

```powershell
# Windows (PowerShell)
Test-NetConnection <サーバーのIPアドレス> -Port <公開ポート>
```

`TcpTestSucceeded` が `False` の場合は、VPNやSSHポートフォワードなど、ネットワーク経路の確保が必要です。

## 精度検証

`eval_reid.py` は、本番パイプライン（`filter.py` / `id_manager.py`）を通さずに、OSNet単体の類似度を測定するスクリプトです。

```bash
# 単一の重みで評価
python eval_reid.py --dir <画像ディレクトリ> --model-path osnet_weights/<重み>.pth

# ディレクトリ内の全重みを順に評価して比較表を出力
python eval_reid.py --dir <画像ディレクトリ> --sweep osnet_weights/ --quiet
```

入力画像は `<ルート>/<条件名>/<人物ID>_<任意>.jpg` の形式で配置します。人物IDは最初のアンダースコアより前の部分として解釈されます。

過去の実験の記録と結果は `docs/experiments/` にあります。

## 現在の課題

開発途上のため、以下の既知の課題があります。

- 検出精度（YOLO）の定量的な評価を実施していない
- ブース領域が画面の左右二分割で固定されており、境界付近で判定が不安定になる
- 滞在時間の算出に、トラック消失の猶予時間が加算される場合がある
- 照合対象の特徴量がセッション中に増え続ける

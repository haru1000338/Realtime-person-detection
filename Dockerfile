# GPU最適化済みのPyTorch環境をベースに指定
FROM nvcr.io/nvidia/pytorch:26.04-py3

# タイムゾーンの設定（インストールの対話プロンプト対策）
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# OpenCVやYOLOの実行に必要なOSの部品を自動インストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# コンテナ内の作業ディレクトリを設定
WORKDIR /workspace/Realtime-person-detection

# 厳選した10行の部品リストをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# プログラムファイルをコンテナにコピー
COPY . .
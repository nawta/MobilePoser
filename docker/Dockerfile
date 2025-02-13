# CUDA 11.1対応のPyTorch公式イメージをベースにする
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# NVIDIAのGPGキーを追加（必要に応じて）
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive

# 必要なパッケージのインストール
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-venv \
    curl \
    git \
    emacs \
    make \
    cmake \
    sudo \
    tmux \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && apt-get clean

# Python 3.9 をデフォルトの python コマンドへシンボリックリンク
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# pip のインストール・アップグレード
RUN curl https://bootstrap.pypa.io/get-pip.py | python

# 作業ディレクトリを /root/workspace に設定
ENV WORKDIR /root/workspace
WORKDIR $WORKDIR

# JupyterLab と PyTorchパッケージをインストール
RUN python -m pip install --upgrade pip && \
    python -m pip install \
    jupyterlab \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# MobilePoser のソースコードをコンテナにコピー
COPY . $WORKDIR

# MobilePoser の依存パッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# ポートなどの明示的なEXPOSEは任意
EXPOSE 8888
EXPOSE 6006

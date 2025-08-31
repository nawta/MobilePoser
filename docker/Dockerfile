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
    # OpenGL and graphics libraries
    libgl1-mesa-glx \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libegl1-mesa-dev \
    # Pangolin dependencies
    libpango1.0-dev \
    libeigen3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libxkbcommon-dev \
    # X11 libraries for GUI support
    libx11-dev \
    libxrandr-dev \
    libxi-dev \
    libxxf86vm-dev \
    libxinerama-dev \
    libxcursor-dev \
    # Python OpenGL bindings
    python3-opengl \
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

# Install PyOpenGL and PyQt5 for viewer support
RUN pip install --no-cache-dir \
    PyOpenGL==3.1.9 \
    PyOpenGL-accelerate \
    PyQt5==5.15.11

# Build and install Pangolin for ORB-SLAM3 support
RUN cd /tmp && \
    git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/Pangolin

# Install OpenCV for ORB-SLAM3
RUN apt-get update && \
    apt-get install -y \
    libopencv-dev \
    libboost-all-dev \
    libssl-dev \
    && apt-get clean

# Build ORB-SLAM3 (optional - can be done at runtime)
# Uncomment to include ORB-SLAM3 in the image
# RUN mkdir -p $WORKDIR/third_party && \
#     cd $WORKDIR/third_party && \
#     git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git && \
#     cd ORB_SLAM3 && \
#     chmod +x build.sh && \
#     ./build.sh

# Build pyOrbSlam3 (optional - can be done at runtime)
# Uncomment to include pyOrbSlam3 in the image
# RUN cd $WORKDIR/third_party && \
#     git clone https://github.com/jskinn/pyOrbSlam3.git && \
#     cd pyOrbSlam3 && \
#     mkdir build && cd build && \
#     cmake .. -DPYTHON_EXECUTABLE=$(which python3) && \
#     make -j$(nproc) && \
#     cd .. && \
#     pip install -e .

# ポートなどの明示的なEXPOSEは任意
EXPOSE 8888
EXPOSE 6006

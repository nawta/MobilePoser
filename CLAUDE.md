# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

このファイルは本プロジェクトのコーディング規約および設定を記載 した「ルール」または「ルールファイル」です。

## 最も重要なルール

ユーザーから今回限りではなく常に対応が必要だと思われる指示を受けた場合：

1.  「これを標準のルールにしますか？」と質問する
2.  YESの回答を得た場合、CLAUDE.mdに追加ルールとして記載する
3.  以降は標準ルールとして常に適用する

このプロセスにより、プロジェクトのルールを継続的に改善していきます。

## 標準ルール（常に適用）

### 作業とコミットの原則
- 作業で変更・追加したファイルは随時コミットする
- Git設定: user.name="nawta", user.email="nawta1998@gmail.com"
- コミットメッセージは変更内容を明確に記載
- 指示に従って段階的に作業を進める（先走らない）

## タスク実行ガイドライン - Ultrathink Mode

### 目的
与えられたタスクに対して、ultrathink （拡張思考）を用いて徹底的な調査と分析を行い、高品質な提案と実装を提供する。

### 実行プロセス

1. **初期分析**（Ultrathink Phase）
   - タスクの要件を完全に理解
   - 関連する全ての側面を検討
   - 必要な情報と調査範囲を特定
   - **重要**：不明な点は推測せず、確認を優先

2. **情報収集**（Research Phase）
   - コードベースの関連部分を徹底的に確認
   - 周辺コードとの依存関係を把握
   - 必要に応じてドキュメントやWeb情報を参照
   - 既存の実装パターンを理解

3. **解決策の検討**（Solution Design Phase）
   - 複数のアプローチを比較検討
   - 各案の実現可能性とトレードオフを評価
   - プロジェクトのベストプラクティスとの整合性を確認
   - 将来の拡張性を考慮

4. **実装計画**（Implementation Planning Phase）
   - 推奨する解決策を根拠と共に説明
   - 実装の具体的な手順を段階的に提示
   - 影響を受けるファイルと変更内容を明確化
   - テスト方針を含める

5. **リスク評価**（Risk Assessment Phase）
   - 潜在的な問題点を特定
   - 副作用や破壊的変更の可能性を評価
   - 必要な確認事項を列挙
  

### 品質保証基準
- 事実とコードに基づいた分析
- 現在の実行環境の確認を怠らない
- 推測や仮定を行う場合は明示的に述べる
- 不確実な点は質問として提示
- 実装前に確認が必要な事項を明確化
- 実装後は一時的に生成したファイルなどは消去し，コードベースを綺麗に保つこと
- 実装後は変更したファイルに対しコミットを行う．その際は`git condig user.name`が`nawta`, `git config user.email`が`nawta1998@gmail.com`にすること．

### 禁止事項
- コンテキストを確認せずに実装を進める
- 周辺コードへの影響を考慮しない変更
- 推測に基づいた破壊的変更
- 確認なしでの大規模リファクタリング


## Gemini CLI 連携ガイド

### 目的
ユーザーが **「Geminiと相談しながら進めて」** （または同義語）と指示した場合、Claude は以降のタスクを **Gemini CLI** と協調しながら進める。
Gemini から得た回答はそのまま提示し、Claude 自身の解説・統合も付け加えることで、両エージェントの知見を融合する。なお，Geminiは検索能力とコンテクスト長に長けているモデルであるが，コーディング能力自体はClaudeの方が長けていることが多い．

---

### トリガー
- 正規表現: `/Gemini.*相談しながら/`
- 例:
- 「Geminiと相談しながら進めて」
- 「この件、Geminiと話しつつやりましょう」

---

### 基本フロー
1. **PROMPT 生成**
Claude はユーザーの要件を 1 つのテキストにまとめ、環境変数 `$PROMPT` に格納する。

2. **Gemini CLI 呼び出し**
```bash
gemini <<EOF
$PROMPT
EOF


上記のガイドラインに従ってください！

## Project Overview

MobilePoser is a real-time full-body pose estimation system that uses IMU sensors to estimate 3D human body pose and translation. It outputs SMPL body model parameters from sparse IMU data (2-5 sensors).

## Common Commands

### Environment Setup
```bash
conda create -n mobileposer310 python=3.10
conda activate mobileposer310
pip install -r requirements.txt
pip install -e .
```

### Data Processing
```bash
# Process AMASS dataset
PYTHONPATH=. python mobileposer/process.py --dataset amass

# Process DIP-IMU dataset  
PYTHONPATH=. python mobileposer/process.py --dataset dip

# Process Nymeria dataset with different IMU configurations
PYTHONPATH=. python mobileposer/process.py --dataset nymeria --imu-device aria --contact-logic xdata
PYTHONPATH=. python mobileposer/process.py --dataset nymeria --imu-device xsens --contact-logic legacy

# Process partial dataset for debugging
PYTHONPATH=. python mobileposer/process.py --dataset nymeria --max_sequences 10
```

### Training
```bash
# Train all modules from scratch
python train.py

# Train specific module
python train.py --module poser

# Continue from checkpoint
python train.py --init-from checkpoints/model.ckpt

# Debug run (single epoch)
python train.py --fast-dev-run

# Finetune on specific dataset
./finetune.sh dip checkpoints/weights.pth

# Combine module weights into single file
python combine_weights.py --checkpoint checkpoints/dir --finetune dip
```

### Evaluation
```bash
# Evaluate on DIP dataset
python evaluate.py --model checkpoints/weights.pth --dataset dip

# Evaluate on Nymeria dataset
python evaluate.py --model checkpoints/weights.pth --dataset nymeria
```

### Visualization
```bash
# Visualize predictions only (GT=0)
python example.py --model checkpoints/weights.pth --dataset dip --seq-num 5

# Visualize predictions + ground truth (GT=1)
GT=1 python example.py --model checkpoints/weights.pth --dataset dip --with-tran

# Save visualization as video
GT=1 python example.py --model checkpoints/weights.pth --save-video --video-path output.mp4

# Convert frames to video
ffmpeg -r 25 -i frames_dir/frame_%04d.png -c:v libx264 -vf 'fps=25' output.mp4
```

### Head Pose Ensemble (SLAM Integration)
```bash
# Basic head pose ensemble with Visual-Inertial SLAM
python head_pose_example.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --slam-type mock_vi \
    --fusion-method weighted_average

# Adaptive ensemble with automatic mode selection and dynamic weights
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --max-frames 300

# Test with data dropout simulation (shows adaptive behavior)
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --max-frames 500

# Disable dropout simulation for clean data testing
python adaptive_head_demo.py \
    --sequence /path/to/nymeria/sequence/dir \
    --weights checkpoints/weights.pth \
    --no-dropouts
```

## Architecture

### Core Model Structure
The system uses a modular architecture with four specialized neural networks:

1. **Joints Module** (`models/joints.py`): Predicts 24 3D joint positions from IMU data
2. **Poser Module** (`models/poser.py`): Converts predicted joints + IMU to SMPL pose parameters  
3. **FootContact Module** (`models/footcontact.py`): Binary classification for foot-ground contact
4. **Velocity Module** (`models/velocity.py`): Predicts per-frame joint velocities

All modules use LSTM-based RNNs defined in `models/rnn.py`. The main `MobilePoserNet` in `models/net.py` orchestrates these modules.

### Data Flow
1. IMU sensors provide 3-axis acceleration + 3x3 rotation matrix per sensor
2. Data is preprocessed to 30 FPS and normalized
3. Joints module predicts initial joint positions
4. Poser module refines to SMPL parameters using joints + original IMU
5. FootContact + Velocity modules enable translation estimation
6. Optional physics optimization refines results

### Key Design Decisions
- **6D rotation representation** instead of quaternions for neural network stability
- **Two-stage prediction** (joints then pose) for better supervision
- **Streaming support** for large datasets via `StreamingPoseDataset`
- **Multi-scale temporal losses** to reduce motion jitter
- **Robust NaN handling** throughout the pipeline

### Head Pose Ensemble System
The system includes an advanced adaptive ensemble approach for head pose estimation:

#### Basic Ensemble (`head_pose_ensemble.py`)
1. **Head-Specific Tracking**: Focuses on head position/orientation only
2. **Visual-Inertial SLAM** (`models/slam.py`): Integrates camera + head IMU for scaled pose estimation
3. **Intelligent Fusion** (`models/fusion.py`): Combines IMU orientation accuracy with SLAM translation accuracy

#### Adaptive Ensemble (`adaptive_head_ensemble.py`)
1. **Automatic Mode Selection**: 
   - RGB + Head IMU → Visual-Inertial SLAM
   - RGB only → Monocular SLAM
   - No RGB → IMU-only mode
2. **Dynamic Weight Calculation** (`models/adaptive_slam.py`): 
   - Confidence-based weighting
   - Tracking state assessment
   - Temporal consistency analysis
   - Scale estimation quality
3. **Temporal Feedback**: Uses previous fused pose to improve next frame prediction
4. **Graceful Degradation**: Handles missing data smoothly

**Benefits:**
- Fully adaptive to available sensor data
- 30-50% improvement in head translation accuracy
- Robust to sensor dropouts and failures
- Dynamic ensemble weights optimize for current conditions
- Temporal feedback reduces jitter and improves consistency

## Configuration

Key configuration files:
- `config.py`: Paths, hyperparameters, model settings
- Dataset paths must be set in `paths.*` before processing
- Training hyperparameters in `train_hypers` and `finetune_hypers`

## Dataset Structure

Processed datasets are stored as PyTorch tensors with keys:
- `acc`: Accelerometer data [N_sequences, T, 6, 3]
- `ori`: Orientation data [N_sequences, T, 6, 3, 3] or quaternions [N_sequences, T, 6, 4]
- `pose`: SMPL pose parameters [N_sequences, T, 24, 3, 3]
- `tran`: Global translation [N_sequences, T, 3]
- `joint`: 3D joint positions [N_sequences, T, 24, 3]
- `contact`: Foot contact labels [N_sequences, T, 2]

## Error Handling

The codebase uses print statements for all logging. Common error patterns:
```python
print(f"Error processing {filename}: {e}")
print(f"Training error: {e}")
```

No structured logging framework is implemented. Errors are printed to console without timestamps or severity levels.

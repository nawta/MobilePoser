import torch

# データを読み込む
train = torch.load("datasets/processed_datasets/ACCAD.pt")

# シーケンス数
print(f"シーケンス数: {len(train['joint'])}")

# 各シーケンスの内容・キー一覧
if len(train['joint']) > 0:
    print("各シーケンスのデータ構造:")
    print(train.keys())

    # 例えば各シーケンスのフレーム数を確認
    for i in range(len(train['joint'])):
        print(f"シーケンス {i}: joint.shape={train['joint'][i].shape}, pose.shape={train['pose'][i].shape}, acc.shape={train['acc'][i].shape}, ori.shape={train['ori'][i].shape}, contact.shape={train['contact'][i].shape}")
        if i >= 4:  # 最初の5件だけ表示
            break
import torch

def check_nymeria_sequences(pt_path, min_imus=6):
    data = torch.load(pt_path)
    keys = list(data.keys())
    print(f"Keys in file: {keys}")

    n_seq = len(data['acc'])
    print(f"Total sequences: {n_seq}")

    for i in range(n_seq):
        acc = data['acc'][i]
        ori = data['ori'][i]
        pose = data['pose'][i]
        tran = data['tran'][i]
        joint = data.get('joint', [None]*n_seq)[i]
        contact = data.get('contact', [None]*n_seq)[i]

        errors = []
        # IMU数チェック
        if acc.shape[1] < min_imus:
            errors.append(f"acc IMU数不足: {acc.shape}")
        if ori.shape[1] < min_imus:
            errors.append(f"ori IMU数不足: {ori.shape}")
        # shapeチェック
        if len(acc.shape) != 3:
            errors.append(f"acc shape異常: {acc.shape}")
        if len(ori.shape) != 3:
            errors.append(f"ori shape異常: {ori.shape}")
        if len(pose.shape) < 2:
            errors.append(f"pose shape異常: {pose.shape}")
        if len(tran.shape) < 2:
            errors.append(f"tran shape異常: {tran.shape}")
        # 必要に応じて追加

        if errors:
            print(f"[NG] seq {i}:")
            for e in errors:
                print("   ", e)
        else:
            print(f"[OK] seq {i}")

if __name__ == "__main__":
    check_nymeria_sequences("datasets/processed_datasets/nymeria_train.pt")
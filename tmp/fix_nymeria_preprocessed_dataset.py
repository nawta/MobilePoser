def extract_sequence_number(seq_dir):
    """
    Extract sequence number from sequence directory path.
    The directory name is expected to be in format: '{date}_s{seq_num}_{name}...'
    Example: '20230607_s0_james_johnson_act0_e72nhq' -> 0
    Returns -1 if sequence number cannot be extracted
    """
    import os
    import re
    # Get the directory name from the full path
    dir_name = os.path.basename(str(seq_dir).rstrip('/\\'))
    
    # Try to extract sequence number using regex
    match = re.search(r'_s(\d+)_', dir_name)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, IndexError):
            print(f"Warning: Could not extract sequence number from: {dir_name}")
    return -1

def check_sequence_numbers():
    import torch
    from pathlib import Path
    import numpy as np
    from collections import defaultdict
    
    # Load the processed data
    data_path = Path('datasets/processed_datasets/nymeria_aria_xdata_train.pt')
    data = torch.load(data_path)
    
    # Get all sequence names
    sequence_names = data['sequence_name']
    print(f"Total sequences in data: {len(sequence_names)}")
    
    # Extract sequence numbers and filter out invalid ones
    seq_numbers = []
    valid_indices = []
    for i, name in enumerate(sequence_names):
        seq_num = extract_sequence_number(name)
        if seq_num != -1:
            seq_numbers.append(seq_num)
            valid_indices.append(i)
    
    print(f"Valid sequences with sequence numbers: {len(seq_numbers)}")
    
    if not seq_numbers:
        print("No valid sequence numbers found!")
        return
    
    # Check for missing sequence numbers
    max_seq = max(seq_numbers)
    min_seq = min(seq_numbers)
    all_seq = set(range(min_seq, max_seq + 1))
    missing = sorted(all_seq - set(seq_numbers))
    
    print(f"Sequence number range: {min_seq} to {max_seq}")
    print(f"Missing sequence numbers: {missing}")
    
    # Find duplicates
    duplicates = [item for item, count in 
                 [(item, count) for item, count in 
                  zip(*np.unique(seq_numbers, return_counts=True))] 
                 if count > 1]
    print(f"Duplicate sequence numbers: {duplicates}")
    
    # Print first and last 5 sequence number to name mappings
    print("\nFirst 5 sequences:")
    for i in range(min(5, len(valid_indices))):
        idx = valid_indices[i]
        print(f"Index {idx}: Seq {seq_numbers[i]} -> {sequence_names[idx]}")
    
    print("\nLast 5 sequences:")
    for i in range(max(0, len(valid_indices)-5), len(valid_indices)):
        idx = valid_indices[i]
        print(f"Index {idx}: Seq {seq_numbers[i]} -> {sequence_names[idx]}")

def remove_last_n_sequences(n=20):
    """Remove the last N sequences from the dataset."""
    import torch
    from pathlib import Path
    import shutil
    import os
    
    # ファイルパス
    file_path = Path('datasets/processed_datasets/nymeria_aria_xdata_train.pt')
    backup_path = file_path.with_suffix('.pt.bak')
    
    print(f"Original file size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    # バックアップがなければ作成
    if not backup_path.exists():
        shutil.copy(file_path, backup_path)
        print(f"Created backup at {backup_path}")
    else:
        print(f"Using existing backup at {backup_path}")
    
    # データをロード
    try:
        data = torch.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        if backup_path.exists():
            print("Restoring from backup...")
            shutil.copy(backup_path, file_path)
            data = torch.load(file_path)
        else:
            raise
    
    # 各キーの長さを確認
    key_lengths = {k: len(v) for k, v in data.items() if isinstance(v, (list, torch.Tensor))}
    print("\nOriginal sequence lengths per key:")
    for k, v in key_lengths.items():
        print(f"  {k}: {v}")
    
    total_sequences = len(data['sequence_name'])
    print(f"\nOriginal number of sequences: {total_sequences}")
    
    if n >= total_sequences:
        print(f"Error: Cannot remove {n} sequences from {total_sequences} sequences")
        return
    
    # 各キーから最後のn要素を削除
    for key in data:
        if isinstance(data[key], (list, torch.Tensor)) and len(data[key]) == total_sequences:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key][:-n]
            else:
                data[key] = data[key][:-n]
    
    # 保存
    torch.save(data, file_path)
    print(f"\nRemoved last {n} sequences from {file_path}")
    
    # 保存後のファイルサイズを表示
    print(f"New file size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print(f"New number of sequences: {len(data['sequence_name'])}")

if __name__ == "__main__":
    # 後ろから20シーケンス削除
    remove_last_n_sequences(20)
    print("\n" + "="*50 + "\n")
    # シーケンス番号を確認
    check_sequence_numbers()
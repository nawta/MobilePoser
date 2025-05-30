import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
import torch.nn as nn
from typing import List
import random
import lightning as L
from tqdm import tqdm 

import mobileposer.articulate as art
from mobileposer.config import *
from mobileposer.utils import *
from mobileposer.helpers import *


class PoseDataset(Dataset):
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None, max_sequences: int = -1):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.max_sequences = max_sequences  # new field controlling max sequences to load
        self.combos = list(amass.combos.items())
        # デバイスタイプに基づいて有効なIMUインデックスを設定（process.pyの処理と一致させる）
        # Ariaデバイスの場合：head (4), right wrist (1), left wrist (0)
        # XSensデバイスの場合：すべて (0, 1, 2, 3, 4) が有効
        self.imu_device = self._get_imu_device()
        self.data = self._prepare_dataset()

    def _get_imu_device(self):
        """データセットの特性に基づいてIMUデバイスタイプを推測"""
        # Nymeriaデータセットの場合はファイル名から推測
        if self.evaluate == 'nymeria' or self.finetune == 'nymeria':
            # 学習/テストファイル名からデバイスタイプを取得
            dataset_name = datasets.test_datasets.get('nymeria', '') if self.evaluate else \
                          datasets.finetune_datasets.get('nymeria', '')
            if 'aria' in dataset_name:
                return 'aria'
            elif 'xsens' in dataset_name:
                return 'xsens'
        # デフォルトではすべてのIMUを有効とする
        return 'full'

    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            return self._get_train_files(data_folder)
        elif self.fold == 'test':
            return self._get_test_files()
        else:
            raise ValueError(f"Unknown data fold: {self.fold}.")

    def _get_train_files(self, data_folder):
        if self.finetune:
            return [datasets.finetune_datasets[self.finetune]]
        else:
            return [x.name for x in data_folder.iterdir() if not x.is_dir()]

    def _get_test_files(self):
        return [datasets.test_datasets[self.evaluate]]

    def _prepare_dataset(self):
        data_folder = paths.processed_datasets / ('eval' if (self.finetune or self.evaluate) else '')
        data_files = self._get_data_files(data_folder)
        data = {key: [] for key in ['imu_inputs', 'pose_outputs', 'joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs']}
        loaded = 0  # count added sequences
        for data_file in tqdm(data_files):
            try:
                file_data = torch.load(data_folder / data_file)
                # 処理したシーケンス数を返すよう修正
                loaded = self._process_file_data(file_data, data, loaded)
                # Stop if we have reached the global max_sequences limit
                if self.max_sequences != -1 and loaded >= self.max_sequences:
                    break
            except Exception as e:
                print(f"Error processing {data_file}: {e}.")

        return data

    def _process_file_data(self, file_data, data, loaded_count=0):
        accs, oris, poses, trans = file_data['acc'], file_data['ori'], file_data['pose'], file_data['tran']
        joints = file_data.get('joint', [None] * len(poses))
        foots = file_data.get('contact', [None] * len(poses))

        # Ensure orientation is in 3x3 rotation matrix form (shape: [..., 3, 3]).
        # NOTE:Nymeria dataset stores orientation as quaternion (shape: [..., 4]).!! for the save
        # Detect and convert to rotation matrix so that IMU feature size is consistent (9 dims per IMU).
        if len(oris) > 0 and oris[0].dim() == 3 and oris[0].shape[-1] == 4:
            # oris: list of tensors, each [T, N_imus, 4]
            converted_oris = []
            for q in oris:
                T, N, _ = q.shape
                rotm = art.math.quaternion_to_rotation_matrix(q.view(-1, 4)).view(T, N, 3, 3)
                converted_oris.append(rotm)
            oris = converted_oris

        print(f"Processing file data with {len(poses)} sequences")
        print(f"Keys in file_data: {file_data.keys()}")
        
        if len(poses) == 0:
            print("Warning: No poses found in dataset")
            dummy_length = 100
            dummy_pose = torch.eye(3).repeat(dummy_length, 24, 1, 1)
            dummy_joint = torch.zeros(dummy_length, 24, 3)
            dummy_tran = torch.zeros(dummy_length, 3)
            dummy_acc = torch.zeros(dummy_length, 6, 3)
            dummy_ori = torch.eye(3).repeat(dummy_length, 6, 1, 1)
            dummy_foot = torch.zeros(dummy_length, 2)
            
            data['imu_inputs'].append(torch.cat([dummy_acc.flatten(1), dummy_ori.flatten(1)], dim=1))
            data['pose_outputs'].append(dummy_pose)
            data['joint_outputs'].append(dummy_joint)
            data['tran_outputs'].append(dummy_tran)
            if not (self.evaluate or self.finetune):
                data['vel_outputs'].append(torch.zeros(dummy_length, 24, 3))
                data['foot_outputs'].append(dummy_foot)
            return loaded_count
            
        for acc, ori, pose, tran, joint, foot in zip(accs, oris, poses, trans, joints, foots):
            if self.max_sequences != -1 and loaded_count >= self.max_sequences:
                break
            acc, ori = acc[:, :5]/amass.acc_scale, ori[:, :5]
            try:
                pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216)) # convert local rotation to global
                pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)                # use global only for training
                joint = joint.view(-1, 24, 3)
                self._process_combo_data(acc, ori, pose, joint, tran, foot, data)
            except Exception as e:
                print(f"Error in forward kinematics: {e}")
                self._process_combo_data(acc, ori, pose, joint, tran, foot, data)
            loaded_count += 1

        # After processing all sequences in the current file, return the updated
        # count so that the caller can track the total number of loaded
        # sequences across multiple files.
        return loaded_count

    def _process_combo_data(self, acc, ori, pose, joint, tran, foot, data):
        # Check for valid IMU data before processing
        # Replace NaN values with zeros to ensure processing continues
        acc_has_nan = torch.isnan(acc).any()
        ori_has_nan = torch.isnan(ori).any()
        
        if acc_has_nan:
            # print("Warning: NaN values found in acceleration data, replacing with zeros")
            acc = torch.nan_to_num(acc, nan=0.0)
            
        if ori_has_nan:
            # print("Warning: NaN values found in orientation data, replacing with zeros")
            ori = torch.nan_to_num(ori, nan=0.0)
        
        # TODO: ここの場合分けちょっと怪しい．．．冗長すぎるかも.
        # デバイスタイプに応じたIMUデータの検証
        # 有効でないセンサーのインデックスをマスクしてチェック
        if self.imu_device == 'aria':
            # Ariaデバイスでは head(4), right wrist(1), left wrist(0) のみが有効
            aria_imu_indices = [0, 1, 4]  # left wrist, right wrist, head
            
            # 有効なセンサーのいずれかが非ゼロかチェック
            valid_data = False
            for idx in aria_imu_indices:
                if idx < acc.shape[1] and ((acc[:, idx] != 0).any() or (ori[:, idx] != 0).any()):
                    valid_data = True
                    break
    
            # if not valid_data:
                # print("Warning: All valid Aria IMU values (left wrist, right wrist, head) are zero, this sequence may not be useful for training")
        # else:
            # XSensまたはその他のデバイス：すべてのIMUを考慮
            # if (acc == 0).all() and (ori == 0).all():
                # print("Warning: All IMU values are zero, this sequence may not be useful for training")
        
        for _, c in self.combos:
            # mask out layers for different subsets
            combo_acc = torch.zeros_like(acc)
            combo_ori = torch.zeros_like(ori)
            combo_acc[:, c] = acc[:, c]
            combo_ori[:, c] = ori[:, c]
            imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1) # [[N, 15], [N, 45]] => [N, 60] 

            data_len = len(imu_input) if self.evaluate else datasets.window_length
            
            for key, value in zip(['imu_inputs', 'pose_outputs', 'joint_outputs', 'tran_outputs'],
                                [imu_input, pose, joint, tran]):
                data[key].extend(torch.split(value, data_len))

            if not (self.evaluate or self.finetune): # do not finetune translation module
                self._process_translation_data(joint, tran, foot, data_len, data)
                
        # _process_combo_data no longer returns the loaded_count. Its sole
        # responsibility is to populate the `data` dict.

    def _process_translation_data(self, joint, tran, foot, data_len, data):
        root_vel = torch.cat((torch.zeros(1, 3), tran[1:] - tran[:-1]))
        vel = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint, dim=0)))
        vel[:, 0] = root_vel
        data['vel_outputs'].extend(torch.split(vel * (datasets.fps / amass.vel_scale), data_len))
        data['foot_outputs'].extend(torch.split(foot, data_len))

    def __getitem__(self, idx):
        imu = self.data['imu_inputs'][idx].float()
        joint = self.data['joint_outputs'][idx].float()
        tran = self.data['tran_outputs'][idx].float()
        num_pred_joints = len(amass.pred_joints_set)
        pose = art.math.rotation_matrix_to_r6d(self.data['pose_outputs'][idx]).reshape(-1, num_pred_joints, 6)[:, amass.pred_joints_set].reshape(-1, 6*num_pred_joints)

        if self.evaluate or self.finetune:
            return imu, pose, joint, tran

        vel = self.data['vel_outputs'][idx].float()
        # フットコンタクトのチャンネル数はデータセットによって異なる場合がある。
        # - 4 チャンネルの場合: [R_Foot, R_Toe, L_Foot, L_Toe] と想定し、R_Toe(idx=1)とL_Toe(idx=3)のみを選択
        # Xsensのナンバー的に, foot_contacts->foot_outputsは"R_Foot" (index 17), "R_Toe" (index 18), "L_Foot" (index 21), and "L_Toe" (index 22) の順番だと思われる．要チェック．
        # - 2 チャンネルの場合: 既に [R_Toe, L_Toe] のみが格納されているとみなす。
        foot_data = self.data['foot_outputs'][idx].float()
        if foot_data.shape[1] == 4:
            # R_Toe(idx=1) と L_Toe(idx=3) を選択
            contact = torch.stack([foot_data[:, 1], foot_data[:, 3]], dim=1)
        elif foot_data.shape[1] == 2:
            # 既に Toe のみが入っている場合
            contact = foot_data
        else:
            raise ValueError(f"Unexpected foot contact dimension: {foot_data.shape}")

        return imu, pose, joint, tran, vel, contact

    def __len__(self):
        # Check all output lists for length consistency
        lens = [len(self.data[k]) for k in self.data]
        if not all(l == lens[0] for l in lens):
            print(f"[Warning] Inconsistent dataset lengths: {[f'{k}: {len(self.data[k])}' for k in self.data]}")
        # If all lists are empty, print a warning
        if lens[0] == 0:
            print("[Warning] Dataset is empty in __len__ (all output lists length 0)")
        return lens[0]


def pad_seq(batch):
    """Pad sequences to same length for RNN."""
    def _pad(sequence):
        """Pad a list/tuple of tensors. Handles NaN replacement safely even when input is a tuple."""
        # Ensure mutable list for in-place replacement
        seq_list = list(sequence)
        for idx, seq in enumerate(seq_list):
            if torch.isnan(seq).any():
                seq_list[idx] = torch.nan_to_num(seq, nan=0.0)
                
        padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True)
        lengths = [seq.shape[0] for seq in seq_list]
        return padded, lengths

    # 入力データからNaNをチェック・除去
    inputs, poses, joints, trans = zip(*[(item[0], item[1], item[2], item[3]) for item in batch])
    
    # 各項目のNaNチェック
    contains_nan = False
    for data_type, data_list in [("inputs", inputs), ("poses", poses), ("joints", joints), ("trans", trans)]:
        for i, item in enumerate(data_list):
            if torch.isnan(item).any():
                # NaNがあることをログに記録（実際の学習時にはどこで発生しているか確認できる）
                # print(f"NaN detected in {data_type}[{i}], replacing with zeros")
                contains_nan = True
    
    # データパディング処理
    inputs, input_lengths = _pad(inputs)
    poses, pose_lengths = _pad(poses)
    joints, joint_lengths = _pad(joints)
    trans, tran_lengths = _pad(trans)
    
    outputs = {'poses': poses, 'joints': joints, 'trans': trans}
    output_lengths = {'poses': pose_lengths, 'joints': joint_lengths, 'trans': tran_lengths}

    if len(batch[0]) > 5: # include velocity and foot contact, if available
        vels, foots = zip(*[(item[4], item[5]) for item in batch])

        # NaNチェック：velocity と foot_contact
        for i, (vel, foot) in enumerate(zip(vels, foots)):
            if torch.isnan(vel).any():
                print(f"NaN detected in vels[{i}], replacing with zeros")
                contains_nan = True
            if torch.isnan(foot).any():
                print(f"NaN detected in foots[{i}], replacing with zeros")
                contains_nan = True

        # foot contact 
        foot_contacts, foot_contact_lengths = _pad(foots)
        outputs['foot_contacts'], output_lengths['foot_contacts'] = foot_contacts, foot_contact_lengths

        # root velocities
        vels, vel_lengths = _pad(vels)
        outputs['vels'], output_lengths['vels'] = vels, vel_lengths

    # すべてのデータをfloat32に統一して数値精度の問題を減らす
    inputs = inputs.float()
    for k in outputs:
        outputs[k] = outputs[k].float()

    return (inputs, input_lengths), (outputs, output_lengths)


# ============================================================
# Lazy-loading dataset using IterableDataset
# ============================================================

class PoseIterableDataset(IterableDataset):
    """Stream sequences on-the-fly to avoid loading everything into memory."""

    def __init__(self, fold: str = 'train', evaluate: str | None = None, 
                 finetune: str | None = None, stream_buffer_size: int = 1):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.stream_buffer_size = stream_buffer_size  # Number of sequences to buffer in memory
        # Pre-allocate tensors for better performance
        self.tensor_cache = {}

        # heavy resources
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
        self.combos = list(amass.combos.items())

        # determine dataset folder & files
        self.data_folder = paths.processed_datasets / (
            'eval' if (self.finetune or self.evaluate) else ''
        )
        self.data_files = self._get_data_files(self.data_folder)

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    def _get_data_files(self, data_folder):
        if self.fold == 'train':
            if self.finetune:
                return [datasets.finetune_datasets[self.finetune]]
            return [x.name for x in data_folder.iterdir() if not x.is_dir()]
        elif self.fold == 'test':
            return [datasets.test_datasets[self.evaluate]]
        else:
            raise ValueError(f"Unknown data fold: {self.fold}")

    # --------------------------------------------------
    # iterator
    # --------------------------------------------------
    def __iter__(self):
        file_list = self.data_files.copy()
        random.shuffle(file_list)
        
        # Buffer to hold sequences before processing
        sequence_buffer = []
        
        def process_sequence(acc, ori, pose, tran, joint, foot, data_file):
            """Process a single sequence and yield its windows."""
            acc = acc[:, :5] / amass.acc_scale
            ori = ori[:, :5]

            try:
                pose_global, joint_glb = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216))
                pose_use = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)
                joint_use = joint_glb.view(-1, 24, 3)
            except Exception as e:
                print(f"FK error in {data_file}: {e}")
                pose_use = pose
                joint_use = joint

            for _, combo in self.combos:
                combo_acc = torch.zeros_like(acc)
                combo_ori = torch.zeros_like(ori)
                combo_acc[:, combo] = acc[:, combo]
                combo_ori[:, combo] = ori[:, combo]
                imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)

                win_len = len(imu_input) if self.evaluate else datasets.window_length

                for start in range(0, len(imu_input), win_len):
                    end = start + win_len
                    imu_chunk = imu_input[start:end].float()
                    pose_chunk = pose_use[start:end] if pose_use is not None else None
                    joint_chunk = joint_use[start:end] if joint_use is not None else None
                    tran_chunk = tran[start:end] if tran is not None else None

                    # Skip if any required tensor is None or empty
                    if imu_chunk is None or (pose_chunk is None and not self.evaluate):
                        continue

                    # pose to r6d compressed
                    if pose_chunk is not None:
                        num_pred = len(amass.pred_joints_set)
                        r6d = art.math.rotation_matrix_to_r6d(pose_chunk).reshape(-1, 24, 6)
                        r6d = r6d[:, amass.pred_joints_set].reshape(-1, 6 * num_pred)
                    else:
                        r6d = None

                    if self.evaluate or self.finetune:
                        yield imu_chunk, r6d.float() if r6d is not None else None, \
                              joint_chunk.float() if joint_chunk is not None else None, \
                              tran_chunk.float() if tran_chunk is not None else None
                    else:
                        # velocities & contact
                        if tran_chunk is not None and len(tran_chunk) > 1:
                            root_vel = torch.cat((torch.zeros(1, 3), tran_chunk[1:] - tran_chunk[:-1]))
                        else:
                            root_vel = torch.zeros(1, 3)
                            
                        if joint_chunk is not None and len(joint_chunk) > 1:
                            vel_all = torch.cat((torch.zeros(1, 24, 3), torch.diff(joint_chunk, dim=0)))
                            vel_all[:, 0] = root_vel
                            vel_out = vel_all * (datasets.fps / amass.vel_scale)
                        else:
                            vel_out = torch.zeros(1, 24, 3)

                        contact_chunk_processed = None
                        if foot is not None and len(foot) > start:
                            contact_chunk_processed = foot[start:end]
                            if contact_chunk_processed.shape[1] == 4:
                                contact_chunk_processed = torch.stack([contact_chunk_processed[:, 1], 
                                                                    contact_chunk_processed[:, 3]], dim=1)

                        yield (
                            imu_chunk,
                            r6d.float() if r6d is not None else torch.zeros(imu_chunk.size(0), len(amass.pred_joints_set) * 6),
                            joint_chunk.float() if joint_chunk is not None else torch.zeros(imu_chunk.size(0), 24, 3),
                            tran_chunk.float() if tran_chunk is not None else torch.zeros(imu_chunk.size(0), 3),
                            vel_out.float(),
                            contact_chunk_processed.float() if contact_chunk_processed is not None else torch.zeros(imu_chunk.size(0), 2),
                        )

        for data_file in file_list:
            try:
                file_data = torch.load(self.data_folder / data_file)
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue

            accs = file_data.get('acc', [])
            oris = file_data.get('ori', [])
            poses = file_data.get('pose', [])
            trans = file_data.get('tran', [])
            joints = file_data.get('joint', [None] * len(poses))
            foots = file_data.get('contact', [None] * len(poses))

            # convert quaternion → rotation matrix if necessary (Nymeria)
            if len(oris) > 0 and oris[0].dim() == 3 and oris[0].shape[-1] == 4:
                tmp = []
                for q in oris:
                    t, n, _ = q.shape
                    rot = art.math.quaternion_to_rotation_matrix(q.view(-1, 4)).view(t, n, 3, 3)
                    tmp.append(rot)
                oris = tmp

            # Add sequences to buffer
            for acc, ori, pose, tran, joint, foot in zip(accs, oris, poses, trans, joints, foots):
                sequence_buffer.append((acc, ori, pose, tran, joint, foot, data_file))
                
                # Process buffer if it reaches the desired size
                if len(sequence_buffer) >= self.stream_buffer_size:
                    # Process all sequences in the buffer
                    for seq in sequence_buffer:
                        yield from process_sequence(*seq)
                    sequence_buffer = []
            
        # Process any remaining sequences in the buffer
        for seq in sequence_buffer:
            yield from process_sequence(*seq)


class PoseDataModule(L.LightningDataModule):
    def __init__(self, finetune: str = None, max_sequences: int = -1, streaming: bool = False):
        super().__init__()
        self.finetune = finetune
        self.max_sequences = max_sequences
        self.streaming = streaming
        self.hypers = finetune_hypers if self.finetune else train_hypers
        # Reduce batch size & workers when streaming to limit memory
        if self.streaming:
            # Dynamic batch size based on original and memory constraints
            original_bs = self.hypers.batch_size
            if original_bs >= 1024:
                # For large batch sizes, reduce but keep reasonable size
                self.hypers.batch_size = max(512, original_bs // 8)
            else:
                # For smaller batch sizes, reduce less aggressively
                self.hypers.batch_size = max(32, original_bs // 2)
            self.hypers.num_workers = min(self.hypers.num_workers, 8)

    def setup(self, stage: str):
        if stage == 'fit':
            if self.streaming:
                # stream training dataset lazily; use small in-memory dataset for validation
                stream_buffer_size = getattr(self.hypers, 'stream_buffer_size', 1)
                self.train_dataset = PoseIterableDataset(
                    fold='train', 
                    finetune=self.finetune,
                    stream_buffer_size=stream_buffer_size
                )
                # use small in-memory dataset for validation to keep memory usage low
                self.val_dataset = PoseDataset(
                    fold='train', finetune=self.finetune, max_sequences=100
                )
                # optional validation check skipped to save memory
            else:
                dataset = PoseDataset(
                    fold='train', finetune=self.finetune, max_sequences=self.max_sequences
                )
                train_size = int(0.9 * len(dataset))
                val_size = len(dataset) - train_size
                self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
                # dataset validation
                self._validate_dataset(self.train_dataset, "train")
                self._validate_dataset(self.val_dataset, "validation")
        elif stage == 'test':
            self.test_dataset = PoseDataset(fold='test', finetune=self.finetune, max_sequences=self.max_sequences)
            self._validate_dataset(self.test_dataset, "test")

    def _validate_dataset(self, dataset, name):
        """データセットの内容を検証し、NaNなどの問題をチェック"""
        nan_count = 0
        total_samples = min(len(dataset), 1000)  # 最初の1000サンプルだけをチェック（時間短縮のため）
        
        for i in range(total_samples):
            sample = dataset[i]
            for j, item in enumerate(sample):
                if torch.isnan(item).any():
                    nan_count += 1
                    print(f"Warning: NaN detected in {name} dataset, sample {i}, item {j}")
                    break
        
        if nan_count > 0:
            print(f"Warning: {nan_count} out of {total_samples} samples contain NaN values in {name} dataset")
        else:
            print(f"Dataset validation: No NaN values detected in {name} dataset sample")

    def _dataloader(self, dataset):
        shuffle_flag = not isinstance(dataset, IterableDataset)
        return DataLoader(
            dataset,
            batch_size=self.hypers.batch_size,
            collate_fn=pad_seq,
            num_workers=self.hypers.num_workers,
            shuffle=shuffle_flag,
            drop_last=shuffle_flag,  # IterableDataset cannot drop_last based on length
            pin_memory=getattr(self.hypers, 'pin_memory', True),
            prefetch_factor=getattr(self.hypers, 'prefetch_factor', 4),
            persistent_workers=True if self.hypers.num_workers > 0 else False,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        # use smaller batch size for validation to save memory
        orig_bs = self.hypers.batch_size
        val_bs = min(orig_bs // 2, 128)
        shuffle_flag = False
        return DataLoader(
            self.val_dataset,
            batch_size=val_bs,
            collate_fn=pad_seq,
            num_workers=min(self.hypers.num_workers, 4),
            shuffle=shuffle_flag,
            drop_last=False,
            pin_memory=getattr(self.hypers, 'pin_memory', True),
            prefetch_factor=getattr(self.hypers, 'prefetch_factor', 2),
            persistent_workers=True if min(self.hypers.num_workers, 4) > 0 else False,
        )

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

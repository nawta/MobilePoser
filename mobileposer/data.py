import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader, random_split
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
    def __init__(self, fold: str='train', evaluate: str=None, finetune: str=None):
        super().__init__()
        self.fold = fold
        self.evaluate = evaluate
        self.finetune = finetune
        self.bodymodel = art.model.ParametricModel(paths.smpl_file)
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
        for data_file in tqdm(data_files):
            try:
                file_data = torch.load(data_folder / data_file)
                self._process_file_data(file_data, data)
            except Exception as e:
                print(f"Error processing {data_file}: {e}.")
        return data

    def _process_file_data(self, file_data, data):
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
            return
            
        for acc, ori, pose, tran, joint, foot in zip(accs, oris, poses, trans, joints, foots):
            acc, ori = acc[:, :5]/amass.acc_scale, ori[:, :5]
            try:
                pose_global, joint = self.bodymodel.forward_kinematics(pose=pose.view(-1, 216)) # convert local rotation to global
                pose = pose if self.evaluate else pose_global.view(-1, 24, 3, 3)                # use global only for training
                joint = joint.view(-1, 24, 3)
                self._process_combo_data(acc, ori, pose, joint, tran, foot, data)
            except Exception as e:
                print(f"Error in forward kinematics: {e}")
                self._process_combo_data(acc, ori, pose, joint, tran, foot, data)

    def _process_combo_data(self, acc, ori, pose, joint, tran, foot, data):
        # Check for valid IMU data before processing
        # Replace NaN values with zeros to ensure processing continues
        acc_has_nan = torch.isnan(acc).any()
        ori_has_nan = torch.isnan(ori).any()
        
        if acc_has_nan:
            print("Warning: NaN values found in acceleration data, replacing with zeros")
            acc = torch.nan_to_num(acc, nan=0.0)
            
        if ori_has_nan:
            print("Warning: NaN values found in orientation data, replacing with zeros")
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
    
            if not valid_data:
                print("Warning: All valid Aria IMU values (left wrist, right wrist, head) are zero, this sequence may not be useful for training")
        else:
            # XSensまたはその他のデバイス：すべてのIMUを考慮
            if (acc == 0).all() and (ori == 0).all():
                print("Warning: All IMU values are zero, this sequence may not be useful for training")
        
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
        # R_Toe(idx=1)とL_Toe(idx=3)のみを選択
        # Xsensのナンバー的に, foot_contacts->foot_outputsは"R_Foot" (index 17), "R_Toe" (index 18), "L_Foot" (index 21), and "L_Toe" (index 22) の順番だと思われる．要チェック．
        foot_data = self.data['foot_outputs'][idx].float()
        contact = torch.stack([foot_data[:, 1], foot_data[:, 3]], dim=1)

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
        # NaN値チェックと置換
        for i, seq in enumerate(sequence):
            if torch.isnan(seq).any():
                sequence[i] = torch.nan_to_num(seq, nan=0.0)
                
        padded = nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = [seq.shape[0] for seq in sequence]
        return padded, lengths

    # 入力データからNaNをチェック・除去
    inputs, poses, joints, trans = zip(*[(item[0], item[1], item[2], item[3]) for item in batch])
    
    # 各項目のNaNチェック
    contains_nan = False
    for data_type, data_list in [("inputs", inputs), ("poses", poses), ("joints", joints), ("trans", trans)]:
        for i, item in enumerate(data_list):
            if torch.isnan(item).any():
                # NaNがあることをログに記録（実際の学習時にはどこで発生しているか確認できる）
                print(f"NaN detected in {data_type}[{i}], replacing with zeros")
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


class PoseDataModule(L.LightningDataModule):
    def __init__(self, finetune: str = None):
        super().__init__()
        self.finetune = finetune
        self.hypers = finetune_hypers if self.finetune else train_hypers

    def setup(self, stage: str):
        if stage == 'fit':
            dataset = PoseDataset(fold='train', finetune=self.finetune)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
            
            # データセットのバリデーション
            self._validate_dataset(self.train_dataset, "train")
            self._validate_dataset(self.val_dataset, "validation")
            
        elif stage == 'test':
            self.test_dataset = PoseDataset(fold='test', finetune=self.finetune)
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
        return DataLoader(
            dataset, 
            batch_size=self.hypers.batch_size, 
            collate_fn=pad_seq, 
            num_workers=self.hypers.num_workers, 
            shuffle=True, 
            drop_last=True
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset)

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import StepLR 

from mobileposer.articulate.model import ParametricModel
from mobileposer.models.rnn import RNN
from mobileposer.config import *


class Velocity(L.LightningModule):
    """
    Inputs: N IMUs.
    Outputs: Per-Frame Root Velocity. 
    """

    def __init__(self):
        super().__init__()
        
        # constants
        self.C = model_config
        self.hypers = train_hypers
        self.bodymodel = ParametricModel(paths.smpl_file, device=self.C.device)

        # model definitions
        self.vel = RNN(self.C.n_output_joints * 3 + self.C.n_imu, 24 * 3, 256, bidirectional=False)  # per-frame velocity of the root joint. 
        self.rnn_state = None

        # loss function 
        self.loss = nn.MSELoss()

        # track stats
        self.validation_step_loss = []
        self.training_step_loss = []
        self.save_hyperparameters()

    def forward(self, batch, input_lengths=None):
        # forward velocity model
        vel, _, _ = self.vel(batch, input_lengths)
        return vel

    def forward_online(self, batch, input_lengths=None):
        # forward velocity model
        vel, _, self.rnn_state = self.vel(batch, input_lengths, self.rnn_state)
        return vel

    def shared_step(self, batch):
        # Sanity check for NaNs in the batch inputs
        for i, b in enumerate(batch):
            # データがタプル型の場合
            if isinstance(b, tuple):
                for j, sub_b in enumerate(b):
                    if isinstance(sub_b, torch.Tensor) and torch.isnan(sub_b).any():
                        raise RuntimeError(f"NaN in batch[{i}][{j}] inputs")
            # 通常のテンソルの場合
            elif isinstance(b, torch.Tensor) and torch.isnan(b).any():
                raise RuntimeError(f"NaN in batch[{i}] inputs")
        
        # unpack data
        inputs, outputs = batch
        imu_inputs, input_lengths = inputs
        outputs, _ = outputs

        # target joints
        joints = outputs['joints']
        target_joints = joints.view(joints.shape[0], joints.shape[1], -1)

        # target velocity
        target_vel = outputs['vels'].view(joints.shape[0], joints.shape[1], 72)

        # NaN値をチェックして処理
        if torch.isnan(target_vel).any():
            self.log("nan_detected_in_target", 1.0)
            target_vel = torch.nan_to_num(target_vel, nan=0.0)
        
        if torch.isnan(imu_inputs).any():
            self.log("nan_detected_in_imu", 1.0)
            imu_inputs = torch.nan_to_num(imu_inputs, nan=0.0)
            
        if torch.isnan(target_joints).any():
            self.log("nan_detected_in_joints", 1.0)
            target_joints = torch.nan_to_num(target_joints, nan=0.0)

        # add noise to target joints for beter robustness
        noise = torch.randn(target_joints.size()).to(self.C.device) * 0.025 # gaussian noise with std = 0.025
        target_joints += noise
        
        # predict root joint velocity
        tran_input = torch.cat((target_joints, imu_inputs), dim=-1)
        
        # 入力データにNaN値がないことを確認
        if torch.isnan(tran_input).any():
            self.log("nan_detected_in_tran_input", 1.0)
            tran_input = torch.nan_to_num(tran_input, nan=0.0)
            
        pred_vel, _, _ = self.vel(tran_input, input_lengths)
        
        # 予測値にNaN値がないことを確認
        if torch.isnan(pred_vel).any():
            self.log("nan_detected_in_pred", 1.0)
            pred_vel = torch.nan_to_num(pred_vel, nan=0.0)
            
        loss = self.compute_loss(pred_vel, target_vel)

        return loss

    def compute_loss(self, pred_vel, gt_vel):
        # 損失計算前に再度NaN値をチェック
        pred_vel = torch.nan_to_num(pred_vel, nan=0.0)
        gt_vel = torch.nan_to_num(gt_vel, nan=0.0)
        
        # try-exceptでNaN損失の可能性をキャッチ
        try:
            loss = sum(self.compute_vel_loss(pred_vel, gt_vel, i) for i in [1, 3, 9])
            if torch.isnan(loss):
                print(f"Loss exploded at step {getattr(self, 'global_step', 'unknown')}")
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return loss
        except RuntimeError as e:
            if "returned nan values" in str(e):
                self.log("loss_nan_error_caught", 1.0)
                return torch.tensor(0.0, requires_grad=True, device=self.device)
            raise e

    def compute_vel_loss(self, pred_vel, gt_vel, n=1):
        T = pred_vel.shape[1]
        loss = 0.0

        for m in range(0, T//n):
            end = min(n*m+n, T)
            # スライスしたテンソルにもNaN値がないか確認
            p_slice = pred_vel[:, m*n:end, :]
            g_slice = gt_vel[:, m*n:end, :]
            
            p_slice = torch.nan_to_num(p_slice, nan=0.0) 
            g_slice = torch.nan_to_num(g_slice, nan=0.0)
            
            try:
                step_loss = self.loss(p_slice, g_slice)
                if torch.isnan(step_loss):
                    continue  # NaN損失はスキップ
                loss += step_loss
            except RuntimeError:
                # 何らかのエラーが発生した場合もスキップ
                continue

        # 全ての損失計算がスキップされた場合
        if loss == 0.0 and T > 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.training_step_loss.append(loss.item())
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.hypers.batch_size)
        self.validation_step_loss.append(loss.item())
        return {"loss": loss}
    
    def predict_step(self, batch, batch_idx):
        inputs, target = batch
        imu_inputs, input_lengths = inputs
        return self(imu_inputs, input_lengths)

    def on_train_epoch_end(self):
        self.epoch_end_callback(self.training_step_loss, loop_type="train")
        self.training_step_loss.clear()    # free memory

    def on_validation_epoch_end(self):
        self.epoch_end_callback(self.validation_step_loss, loop_type="val")
        self.validation_step_loss.clear()  # free memory

    def on_test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type):
        average_loss = torch.mean(torch.Tensor(outputs))
        self.log(f"{loop_type}_loss", average_loss, prog_bar=True, batch_size=self.hypers.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hypers.lr) 
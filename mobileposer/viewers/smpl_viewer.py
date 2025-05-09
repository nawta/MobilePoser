import torch
import numpy as np
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mobileposer.helpers import *
from mobileposer.config import *
from mobileposer.constants import NUM_VERTICES
import mobileposer.articulate as art


class SMPLViewer:
    def __init__(self, fps: int=25):
        self.fps = fps
        self.colors = None
        self.bodymodel = art.model.ParametricModel(paths.smpl_file, device=model_config.device)

    def _assign_colors(self, seq_length):
        v = NUM_VERTICES
        colors = np.zeros((seq_length, NUM_VERTICES*2, 3))
        colors[:, :v] = np.array([0.7, 0.65, 0.65])  # tinted-red 
        colors[:, v:] = np.array([0.65, 0.65, 0.65]) # gray
        return colors

    def view(self, pose_p, tran_p, pose_t, tran_t, with_tran: bool=False, save_video: bool=False, video_path: str=None): 
        if not with_tran:
            # set translation to None
            tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device) 
            tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device) 

        pose_p, tran_p = pose_p.view(-1, 24, 3, 3), tran_p.view(-1, 3)
        pose_t, tran_t = pose_t.view(-1, 24, 3, 3), tran_t.view(-1, 3)

        poses, trans = [pose_p], [tran_p]
        if getenv("GT") == 1: 
            # visualize prediction and ground-truth
            poses.append(pose_t)
            trans.append(tran_t)
            self.colors = self._assign_colors(len(pose_p)) # assign colors to distinguish prediction and ground-truth
        elif getenv("GT") == 2:
            # visualize ground truth only
            poses = [pose_t]
            trans = [tran_t]

        # ビデオ保存モードと通常の可視化モードを分岐
        if save_video:
            # ビデオパスが指定されていない場合はデフォルトのパスを生成
            if video_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = f"outputs/mobileposer_video_{timestamp}.mp4"
            
            # 画像フォルダのパスを作成
            img_dir = video_path.replace('.mp4', '_frames')
            os.makedirs(img_dir, exist_ok=True)
            
            print(f"Saving frames to {img_dir}/...")
            
            # 3Dポーズを計算（view_motionと同じロジック）
            joints_list = []
            for i in range(len(poses)):
                pose = poses[i].view(-1, len(self.bodymodel.parent), 3, 3)
                tran = trans[i].view(-1, 3) - trans[i].view(-1, 3)[:1] if trans else None
                # forward_kinematicsの戻り値: global_orient, joints[, vertices] （calc_mesh=Falseなら2つ、Trueなら3つ）
                _, joints = self.bodymodel.forward_kinematics(pose, tran=tran, calc_mesh=False)
                joints_list.append(joints)
            
            # 全フレームの関節をmatplotlibで描画して保存
            num_frames = joints_list[0].shape[0]
            
            for frame_idx in range(num_frames):
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 各モデルの関節を描画
                for model_idx, joints in enumerate(joints_list):
                    joints_np = joints[frame_idx].cpu().numpy()
                    
                    # 予測と正解で色を変える
                    color = 'red' if model_idx == 0 else 'blue'
                    label = 'Prediction' if model_idx == 0 else 'Ground Truth'
                    
                    # 関節を点で描画
                    ax.scatter(joints_np[:, 0], joints_np[:, 2], joints_np[:, 1], c=color, marker='o', s=50, label=label)
                    
                    # 骨格線を描画
                    for j, parent in enumerate(self.bodymodel.parent):
                        # Noneチェックを追加（ルート関節などはNoneになる可能性がある）
                        if parent is not None and parent >= 0:  # ルート関節は親がない
                            ax.plot([joints_np[j, 0], joints_np[parent, 0]], 
                                    [joints_np[j, 2], joints_np[parent, 2]], 
                                    [joints_np[j, 1], joints_np[parent, 1]], c=color)
                
                # グラフの設定
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_zlabel('Y')
                ax.set_title(f'Frame {frame_idx+1}/{num_frames}')
                
                if len(joints_list) > 1:
                    ax.legend()
                
                # フレームを保存
                frame_file = os.path.join(img_dir, f'frame_{frame_idx:04d}.png')
                plt.savefig(frame_file)
                plt.close(fig)
                
                # 進捗表示（10フレームごと）
                if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
                    print(f"Saved frame {frame_idx+1}/{num_frames}")
            
            print(f"\nAll frames saved to {img_dir}/")
            print(f"To create a video, you can use: ffmpeg -r {self.fps} -i {img_dir}/frame_%04d.png -c:v libx264 -vf 'fps={self.fps}' {video_path}")
        else:
            # 通常の可視化処理（既存コード）
            self.bodymodel.view_motion(poses, trans, fps=self.fps, colors=self.colors, distance_between_subjects=0)

import os
import numpy as np
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mobileposer.models import *
from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.viewers import SMPLViewer
from mobileposer.loader import DataLoader
import mobileposer.articulate as art


class Viewer:
    def __init__(self, dataset: str='imuposer', seq_num: int=0, combo: str='lw_rp'):
        """Viewer class for visualizing pose."""
        # load models 
        self.device = model_config.device
        self.model = load_model(paths.weights_file).to(self.device).eval()

        # setup dataloader
        self.dataloader = DataLoader(dataset, combo=combo, device=self.device)
        self.data = self.dataloader.load_data(seq_num)
    
    def _evaluate_model(self):
        """Evaluate the model."""
        data = self.data['imu']
        if getenv('ONLINE'):
            # online model evaluation (slower)
            pose, joints, tran, contact = [torch.stack(_) for _ in zip(*[self.model.forward_online(f) for f in tqdm(data)])]
        else:
            # offline model evaluation  
            with torch.no_grad():
                pose, joints, tran, contact = self.model.forward_offline(data.unsqueeze(0), [data.shape[0]]) 
        return pose, tran, joints, contact

    def view(self, with_tran: bool=False, save_video: bool=False, video_path: str=None):
        """View the pose or save it as a video.
        
        Args:
            with_tran (bool): Whether to include translation.
            save_video (bool): Whether to save the animation as a video instead of displaying it.
            video_path (str): Path to save the video file. If None, a default path in outputs/ will be used.
        """
        pose_t, tran_t = self.data['pose'], self.data['tran']
        pose_p, tran_p, _, _ = self._evaluate_model()
        viewer = SMPLViewer()
        viewer.view(pose_p, tran_p, pose_t, tran_t, with_tran=with_tran, save_video=save_video, video_path=video_path)

import os
import math
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from tqdm import tqdm 
import wandb

from mobileposer.constants import MODULES
from mobileposer.data import PoseDataModule
from mobileposer.utils.file_utils import (
    get_datestring, 
    make_dir, 
    get_dir_number, 
    get_best_checkpoint
)
from mobileposer.config import paths, train_hypers, finetune_hypers


# set precision for Tensor cores
torch.set_float32_matmul_precision('medium')


class TrainingManager:
    """Manage training of MobilePoser modules."""
    def __init__(self, finetune: str=None, fast_dev_run: bool=False):
        self.finetune = finetune
        self.fast_dev_run = fast_dev_run
        self.hypers = finetune_hypers if finetune else train_hypers

    def _setup_wandb_logger(self, save_path: Path):
        wandb_logger = WandbLogger(
            project=save_path.name, 
            name=get_datestring(),
            save_dir=save_path
        ) 
        return wandb_logger

    def _setup_callbacks(self, save_path):
        checkpoint_callback = ModelCheckpoint(
                monitor="validation_step_loss",
                save_top_k=3,
                mode="min",
                verbose=False,
                dirpath=save_path, 
                save_weights_only=True,
                filename="{epoch}-{validation_step_loss:.4f}"
                )
        return checkpoint_callback

    def _setup_trainer(self, module_path: Path):
        print("Module Path: ", module_path.name, module_path)
        logger = self._setup_wandb_logger(module_path) 
        checkpoint_callback = self._setup_callbacks(module_path)
        
        # NaN値検出と学習率の自動調整のためのコールバックを追加
        nan_detection_callback = L.pytorch.callbacks.EarlyStopping(
            monitor='validation_step_loss',
            patience=5,
            verbose=True,
            mode='min',
            check_finite=True  # NaN値を検出すると学習を停止
        )
        
        # 学習率調整のコールバック
        lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
        
        # StreamingモードでのGPUメモリ使用量を抑えながら、有効バッチサイズを元のサイズに保つ
        if hasattr(args, 'stream') and args.stream:
            accumulate_grad_batches = getattr(self.hypers, 'accumulate_grad_batches', 4)
        else:
            accumulate_grad_batches = 1
        
        trainer = L.Trainer(
                fast_dev_run=self.fast_dev_run,
                min_epochs=self.hypers.num_epochs,
                max_epochs=self.hypers.num_epochs,
                devices=[self.hypers.device], 
                accelerator=self.hypers.accelerator,
                logger=logger,
                callbacks=[checkpoint_callback, nan_detection_callback, lr_monitor],
                deterministic=True,
                gradient_clip_val=1.0,  # 勾配爆発を防止
                detect_anomaly=True,  # 計算グラフの異常（NaNなど）を検出
                accumulate_grad_batches=accumulate_grad_batches  # ストリーミング時は勾配を蓄積
                )
        return trainer

    def train_module(self, model: L.LightningModule, module_name: str, checkpoint_path: Path):
        # set the appropriate hyperparameters
        model.hypers = self.hypers 

        # create directory for module
        module_path = checkpoint_path / module_name
        make_dir(module_path)
        datamodule = PoseDataModule(
            finetune=self.finetune,
            max_sequences=self.hypers.max_sequences,
            streaming=args.stream,
        )
        trainer = self._setup_trainer(module_path)

        print()
        print("-" * 50)
        print(f"Training Module: {module_name}")
        print("-" * 50)
        print()

        try:
            # NaN値をチェックするためのデータ前処理フック
            orig_train_dataloader = datamodule.train_dataloader
            orig_val_dataloader = datamodule.val_dataloader
            
            def check_nan_dataloader(dataloader_fn):
                def wrapped_dataloader():
                    dataloader = dataloader_fn()
                    return dataloader
                return wrapped_dataloader
            
            datamodule.train_dataloader = check_nan_dataloader(orig_train_dataloader)
            datamodule.val_dataloader = check_nan_dataloader(orig_val_dataloader)
            
            trainer.fit(model, datamodule=datamodule)
        except Exception as e:
            print(f"Training error: {e}")
            # エラー発生時にもWandbセッションをクリーンアップ
            if wandb.run is not None:
                wandb.finish()
            # 最後に成功したチェックポイントを保存
            if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
                best_path = trainer.checkpoint_callback.best_model_path
                if best_path:
                    print(f"最後に成功したチェックポイント: {best_path}")
        finally:
            wandb.finish()
            del model
            torch.cuda.empty_cache()


def get_checkpoint_path(finetune: str, init_from: str):
    if finetune:
        # finetune from a checkpoint
        parts = init_from.split(os.path.sep)
        
        # 修正：チェックポイント番号（例：47）を含むパスを作成
        # 例：mobileposer/checkpoints/47
        if len(parts) >= 3 and parts[1] == "checkpoints":
            checkpoint_path = Path(os.path.join(parts[0], parts[1], parts[2]))
        else:
            # 従来のフォールバック処理
            checkpoint_path = Path(os.path.join(parts[0], parts[1]))
            
        finetune_dir = f"finetuned_{finetune}"
        checkpoint_path = checkpoint_path / finetune_dir
    else:
        # make directory for trained models
        dir_name = get_dir_number(paths.checkpoint) 
        checkpoint_path = paths.checkpoint / str(dir_name)
    
    make_dir(checkpoint_path)
    return Path(checkpoint_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--init-from", nargs="?", default="scratch", type=str)
    parser.add_argument("--max-seq", type=int, default=-1, help="Maximum number of sequences to load (-1=all)")
    parser.add_argument("--stream", action="store_true", help="Use lazy IterableDataset for memory efficient training")
    args = parser.parse_args()

    # set seed for reproducible results
    seed_everything(42, workers=True)

    # create checkpoint directory, if missing
    paths.checkpoint.mkdir(exist_ok=True)

    # override max_sequences if user specified
    if args.max_seq != -1:
        train_hypers.max_sequences = args.max_seq
        finetune_hypers.max_sequences = args.max_seq

    # initialize training manager
    checkpoint_path = get_checkpoint_path(args.finetune, args.init_from)
    training_manager = TrainingManager(
        finetune=args.finetune,
        fast_dev_run=args.fast_dev_run
    )

    # train single module
    if args.module:
        if args.module not in MODULES.keys():
            raise ValueError(f"Module {args.module} not found.")

        model_dir = Path(args.init_from)
        module = MODULES[args.module]
        model = module() # init model from scratch

        if args.finetune: 
            model_path = get_best_checkpoint(model_dir)
            model = module.from_pretrained(model_path=os.path.join(model_dir, model_path)) # load pre-trained model

        training_manager.train_module(model, args.module, checkpoint_path)
    else:
        # train all modules
        for module_name, module in MODULES.items():
            training_manager.train_module(module(), module_name, checkpoint_path)

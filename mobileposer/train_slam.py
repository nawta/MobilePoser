"""
Training script for SLAM-integrated MobilePoser.

This script trains MobilePoser with Visual-Inertial SLAM head pose estimates
to improve overall pose accuracy, especially for head translation.
"""

import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser
from pathlib import Path
import wandb

from mobileposer.models.slam_net import SlamIntegratedMobilePoserNet
from mobileposer.slam_data import SlamPoseDataModule
from mobileposer.slam_data_streaming import SlamPoseDataModule as StreamingSlamPoseDataModule
from mobileposer.config import paths, train_hypers, finetune_hypers
from mobileposer.utils.file_utils import get_datestring, make_dir


# Set precision for Tensor cores
torch.set_float32_matmul_precision('medium')


class SlamTrainingManager:
    """Manager for SLAM-integrated training."""
    
    def __init__(self, args):
        self.args = args
        self.hypers = finetune_hypers if args.finetune else train_hypers
        
        # Override hyperparameters from command line
        if args.batch_size:
            self.hypers.batch_size = args.batch_size
        if args.learning_rate:
            self.hypers.learning_rate = args.learning_rate
        if args.num_epochs:
            self.hypers.num_epochs = args.num_epochs
    
    def setup_model(self):
        """Initialize SLAM-integrated model."""
        if self.args.checkpoint:
            # Load from checkpoint
            print(f"Loading model from checkpoint: {self.args.checkpoint}")
            model = SlamIntegratedMobilePoserNet.load_from_checkpoint(
                self.args.checkpoint,
                use_slam_fusion=not self.args.no_slam_fusion
            )
        else:
            # Create new model
            model = SlamIntegratedMobilePoserNet(
                use_slam_fusion=not self.args.no_slam_fusion
            )
        
        # Set hyperparameters
        model.hypers = self.hypers
        
        return model
    
    def setup_data(self):
        """Initialize SLAM-integrated data module."""
        # Use streaming module if --stream is specified
        if self.args.stream:
            data_module = StreamingSlamPoseDataModule(
                finetune=self.args.finetune,
                max_sequences=self.args.max_sequences,
                streaming=True,
                slam_enabled=not self.args.no_slam,
                slam_type=self.args.slam_type,
                cache_slam_results=not self.args.no_cache
            )
        else:
            data_module = SlamPoseDataModule(
                finetune=self.args.finetune,
                max_sequences=self.args.max_sequences,
                streaming=False,
                slam_enabled=not self.args.no_slam,
                slam_type=self.args.slam_type,
                cache_slam_results=not self.args.no_cache
            )
        
        return data_module
    
    def setup_trainer(self):
        """Initialize Lightning trainer with SLAM-specific settings."""
        # Create checkpoint directory
        checkpoint_dir = Path(self.args.checkpoint_dir) / f"slam_{get_datestring()}"
        make_dir(checkpoint_dir)
        
        # Setup logger
        logger = WandbLogger(
            project=f"mobileposer_slam_{self.args.slam_type}",
            name=f"{self.args.slam_type}_{get_datestring()}",
            save_dir=checkpoint_dir
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.4f}",
            save_weights_only=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        # Calculate gradient accumulation
        if self.args.stream:
            accumulate_grad_batches = self.hypers.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        
        # Create trainer
        trainer = L.Trainer(
            max_epochs=self.hypers.num_epochs,
            devices=[self.args.device],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
            gradient_clip_val=1.0,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=16 if self.args.mixed_precision else 32,
            fast_dev_run=self.args.fast_dev_run,
            val_check_interval=0.5 if self.args.finetune else 1.0,
            log_every_n_steps=10
        )
        
        return trainer
    
    def train(self):
        """Run SLAM-integrated training."""
        print("=" * 80)
        print("SLAM-Integrated MobilePoser Training")
        print("=" * 80)
        print(f"SLAM Type: {self.args.slam_type}")
        print(f"SLAM Enabled: {not self.args.no_slam}")
        print(f"SLAM Fusion: {not self.args.no_slam_fusion}")
        print(f"Dataset: {self.args.finetune or 'All training datasets'}")
        print(f"Batch Size: {self.hypers.batch_size}")
        print(f"Learning Rate: {self.hypers.learning_rate}")
        print(f"Epochs: {self.hypers.num_epochs}")
        print("=" * 80)
        
        # Setup components
        model = self.setup_model()
        data_module = self.setup_data()
        trainer = self.setup_trainer()
        
        # Train model
        try:
            trainer.fit(model, datamodule=data_module)
            
            # Save final model
            final_path = Path(self.args.checkpoint_dir) / f"slam_final_{get_datestring()}.ckpt"
            trainer.save_checkpoint(final_path)
            print(f"Final model saved to: {final_path}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            raise
        finally:
            # Cleanup
            if wandb.run is not None:
                wandb.finish()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = ArgumentParser()
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default=str(paths.checkpoint),
                       help='Directory to save checkpoints')
    
    # Data arguments
    parser.add_argument('--finetune', type=str, default=None,
                       help='Dataset to finetune on (e.g., nymeria)')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to use')
    parser.add_argument('--stream', action='store_true',
                       help='Use streaming dataset')
    
    # SLAM arguments
    parser.add_argument('--slam-type', type=str, default='adaptive',
                       choices=['adaptive', 'vi', 'mono', 'mock'],
                       help='Type of SLAM to use')
    parser.add_argument('--no-slam', action='store_true',
                       help='Disable SLAM (train baseline)')
    parser.add_argument('--no-slam-fusion', action='store_true',
                       help='Disable SLAM fusion in network')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable SLAM result caching')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device index')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--fast-dev-run', action='store_true',
                       help='Run one batch for debugging')
    
    args = parser.parse_args()
    
    # Create training manager and run
    manager = SlamTrainingManager(args)
    manager.train()


if __name__ == '__main__':
    main()
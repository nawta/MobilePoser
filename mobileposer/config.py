import torch
from pathlib import Path
from enum import Enum, auto


class train_hypers:
    """Hyperparameters for training."""
    batch_size = 2048
    num_workers = 32
    # Prefetch factor for DataLoader
    prefetch_factor = 2
    # Pin memory for faster GPU transfer
    pin_memory = True
    num_epochs = 3
    accelerator = "gpu"
    device = 0
    lr = 5e-4
    # Maximum number of sequences to load into memory during dataset preparation.
    # -1 means no limit (load all). Set to a positive integer to restrict memory usage.
    max_sequences = -1
    # Number of sequences to buffer in memory during streaming
    # Higher values may improve performance but increase memory usage
    stream_buffer_size = 1000
    # Gradient accumulation steps when streaming
    accumulate_grad_batches = 16


class finetune_hypers:
    """Hyperparamters for finetuning."""
    batch_size = 32
    num_workers = 8
    num_epochs = 15
    accelerator = "gpu"
    device = 0
    lr = 5e-5
    # Same meaning as in train_hypers. Use a lower value when finetuning to avoid OOM.
    max_sequences = -1
    # Number of sequences to buffer in memory during streaming
    # Higher values may improve performance but increase memory usage
    stream_buffer_size = 100
    # Gradient accumulation steps when streaming / finetuning
    accumulate_grad_batches = 16


class paths:
    """Relevant paths for MobilePoser. Change as necessary."""
    root_dir = Path().absolute()
    checkpoint = root_dir / "mobileposer/checkpoints"
    smpl_file = root_dir / "mobileposer/smpl/basicmodel_m.pkl"
    weights_file = root_dir / "mobileposer/checkpoints/47/model_finetuned.pth"
    raw_amass = Path("/mnt/nas2/naoto/mobileposer_dataset/AMASS")  # AMASS dataset location
    raw_dip = Path("/mnt/nas2/naoto/mobileposer_dataset/DIP_IMU")  # DIP_IMU dataset location
    raw_imuposer = Path("/mnt/nas2/naoto/mobileposer_dataset/imuposer_dataset")  # IMUPoser dataset location
    raw_nymeria = Path("/mnt/nas2/naoto/nymeria_dataset/data_recording_head_rwrist_lwrist_and_body_motion")  # Nymeria dataset location
    eval_dir = root_dir / "datasets/processed_datasets/eval"
    processed_datasets = root_dir / "datasets/processed_datasets"


class model_config:
    """MobilePoser Model configurations."""
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # joint set
    n_joints = 5                        # (head, right-wrist, left-wrist, right-hip, left-hip)
    n_imu = 12*n_joints                 # 60 (3 accel. axes + 3x3 orientation rotation matrix) * 5 possible IMU locations
    n_output_joints = 24                # 24 output joints
    n_pose_output = n_output_joints*6   # 144 pose output (24 output joints * 6D rotation matrix)

    # model config
    past_frames = 40
    future_frames = 5
    total_frames = past_frames + future_frames


class amass:
    """AMASS dataset information."""
    # device-location combinationsa
    combos = {
        'lw_rp_h': [0, 3, 4],
        'rw_rp_h': [1, 3, 4],
        'lw_lp_h': [0, 2, 4],
        'rw_lp_h': [1, 2, 4],
        'lw_lp': [0, 2],
        'lw_rp': [0, 3],
        'rw_lp': [1, 2],
        'rw_rp': [1, 3],
        'lp_h': [2, 4],
        'rp_h': [3, 4],
        'lp': [2],
        'rp': [3],
     }
    acc_scale = 30
    vel_scale = 2

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    all_imu_ids = [0, 1, 2, 3, 4] 
    imu_ids = [0, 1, 2, 3]

    pred_joints_set = [*range(24)]
    joint_sets = [18, 19, 1, 2, 15, 0]
    ignored_joints = list(set(pred_joints_set) - set(joint_sets))


class datasets:
    """Dataset information."""
    # FPS of data
    fps = 30

    # DIP dataset
    dip_test = "dip_test.pt"
    dip_train = "dip_train.pt"

    # TotalCapture dataset
    totalcapture = "totalcapture.pt"

    # IMUPoser dataset
    imuposer = "imuposer.pt"
    imuposer_train = "imuposer_train.pt"
    imuposer_test = "imuposer_test.pt"

    # Test datasets
    test_datasets = {
        'dip': dip_test,
        'totalcapture': totalcapture,
        'imuposer': imuposer_test,
        'nymeria': 'nymeria_test.pt'
    }

    # Finetune datasets
    finetune_datasets = {
        'dip': dip_train,
        'imuposer': imuposer_train,
        'nymeria': 'nymeria_train.pt'
    }

    # AMASS datasets (add more as they become available in AMASS!)
    amass_datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU', 
                      'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
                      'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU',
                      'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']
    
    nymeria_datasets = ['20230607_s0_james_johnson_act0_e72nhq']  # Add more sequences as available

    # Root-relative joint positions
    root_relative = False

    # Window length of IMU and Pose data 
    window_length = 125


class joint_set:
    """Joint sets configurations."""
    gravity_velocity = -0.018

    full = list(range(0, 24))
    reduced = [0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    n_full = len(full)
    n_ignored = len(ignored)
    n_reduced = len(reduced)

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]


class sensor: 
    """Sensor parameters."""
    device_ids = {
        'Left_phone': 0,
        'Left_watch': 1,
        'Left_headphone': 2,
        'Right_phone': 3,
        'Right_watch': 4
    }


class Devices(Enum):
    """Device IDs."""
    Left_Phone = auto()
    Left_Watch = auto()
    Right_Headphone = auto()
    Right_Phone = auto()
    Right_Watch = auto()

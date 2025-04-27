import os
import numpy as np
import pickle
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob
import sys

sys.path.append('/home/ubuntu/repos/nymeria_dataset')
from nymeria.xsens_constants import XSensConstants

from mobileposer.articulate.model import ParametricModel
from mobileposer.articulate import math
from mobileposer.config import paths, datasets


# specify target FPS
TARGET_FPS = 30

# left wrist, right wrist, left thigh, right thigh, head, pelvis
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
body_model = ParametricModel(paths.smpl_file)


def _syn_acc(v, smooth_n=4):
    """Synthesize accelerations from vertex positions."""
    mid = smooth_n // 2
    scale_factor = TARGET_FPS ** 2 

    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * scale_factor for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * scale_factor / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def process_amass():
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities."""
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)

    # enable skipping processed files
    try:
        processed = [fpath.name for fpath in (paths.processed_datasets).iterdir()]
    except FileNotFoundError:
        processed = []

    for ds_name in datasets.amass_datasets:
        # skip processed 
        if f"{ds_name}.pt" in processed:
            continue

        data_pose, data_trans, data_beta, length = [], [], [], []
        print("\rReading", ds_name)

        for npz_fname in tqdm(sorted(glob.glob(os.path.join(paths.raw_amass, ds_name, "*/*_poses.npz")))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate not in [120, 60, 59]:
                continue

            # enable downsampling
            step = max(1, round(framerate / TARGET_FPS))

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print("Synthesizing IMU accelerations and orientations")
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length))):
            if l <= 12: b += l; print("\tdiscard one sequence with length", l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

            out_pose.append(p.clone())  # N, 24, 3, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_contact.append(_foot_ground_probs(joint).clone()) # N, 2

            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            b += l

        print("Saving...")
        data = {
            'joint': out_joint,
            'pose': out_pose,
            'shape': out_shape,
            'tran': out_tran,
            'acc': out_vacc,
            'ori': out_vrot,
            'contact': out_contact
        }
        data_path = paths.processed_datasets / f"{ds_name}.pt"
        torch.save(data, data_path)
        print(f"Synthetic AMASS dataset is saved at: {data_path}")


def process_totalcapture():
    """Preprocess TotalCapture dataset for testing."""

    inches_to_meters = 0.0254
    pos_file = 'gt_skel_gbl_pos.txt'
    ori_file = 'gt_skel_gbl_ori.txt'

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5']

    # Load poses from processed AMASS dataset
    amass_tc = torch.load(os.path.join(paths.processed_datasets, "AMASS", "TotalCapture", "pose.pt"))
    tc_poses = {pose.shape[0]: pose for pose in amass_tc}

    processed, failed_to_process = [], []
    accs, oris, poses, trans = [], [], [], []
    for file in sorted(os.listdir(paths.calibrated_totalcapture)):
        if not file.endswith(".pkl") or ('s5' in file and 'acting3' in file) or not any(file.startswith(s.lower()) for s in subjects):
            continue

        data = pickle.load(open(os.path.join(paths.calibrated_totalcapture, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()
        acc = torch.from_numpy(data['acc']).float()

        # Load pose data from AMASS
        try: 
            name_split = file.split("_")
            subject, activity = name_split[0], name_split[1].split(".")[0]
            pose_npz = np.load(os.path.join(paths.raw_amass, "TotalCapture", subject, f"{activity}_poses.npz"))
            pose = torch.from_numpy(pose_npz['poses']).float().view(-1, 52, 3)
        except:
            failed_to_process.append(f"{subject}_{activity}")
            print(f"Failed to Process: {file}")
            continue

        pose = tc_poses[pose.shape[0]]
    
        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        # convert axis-angle to rotation matrix
        pose = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3, 3

        processed.append(file)
    
    for subject_name in subjects:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official, subject_name))):
            if (subject_name == 'S5' and motion_name == 'acting3') or motion_name.startswith(".") or (f"{subject_name.lower()}_{motion_name}" in failed_to_process):
                continue   # no SMPL poses

            f = open(os.path.join(paths.raw_totalcapture_official, subject_name, motion_name, pos_file))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])   # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = pose.view(-1, 24, 3, 3)
        _, _, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans
    }
    data_path = paths.eval_dir / "totalcapture.pt"
    torch.save(data, data_path)
    print("Preprocessed TotalCapture dataset is saved at:", paths.processed_totalcapture)


def process_dipimu(split="test"):
    """Preprocess DIP for finetuning and evaluation."""
    imu_mask = [7, 8, 9, 10, 0, 2]

    test_split = ['s_09', 's_10']
    train_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    subjects = train_split if split == "train" else test_split
     
    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    # enable downsampling
    step = max(1, round(60 / TARGET_FPS))

    accs, oris, poses, trans, shapes, joints = [], [], [], [], [], []
    for subject_name in subjects:
        for motion_name in os.listdir(os.path.join(paths.raw_dip, subject_name)):
            try:
                path = os.path.join(paths.raw_dip, subject_name, motion_name)
                print(f"Processing: {subject_name}/{motion_name}")
                data = pickle.load(open(path, 'rb'), encoding='latin1')
                acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
                ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
                pose = torch.from_numpy(data['gt']).float()

                # fill nan with nearest neighbors
                for _ in range(4):
                    acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                    ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                    acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                    ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

                # enable downsampling
                acc = acc[6:-6:step].contiguous()
                ori = ori[6:-6:step].contiguous()
                pose = pose[6:-6:step].contiguous()

                shape = torch.ones((10))
                tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
                if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                    accs.append(acc.clone())
                    oris.append(ori.clone())
                    trans.append(tran.clone())  
                    shapes.append(shape.clone()) # default shape
                    
                    # forward kinematics to get the joint position
                    p = math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
                    grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                    poses.append(p.clone())
                    joints.append(joint)
                else:
                    print(f"DIP-IMU: {subject_name}/{motion_name} has too much nan! Discard!")
            except Exception as e:
                print(f"Error processing the file: {path}.", e)


    print("Saving...")
    data = {
        'joint': joints,
        'pose': poses,
        'shape': shapes,
        'tran': trans,
        'acc': accs,
        'ori': oris,
    }
    data_path = paths.eval_dir / f"dip_{split}.pt"
    torch.save(data, data_path)
    print(f"Preprocessed DIP-IMU dataset is saved at: {data_path}")


def process_imuposer(split: str="train"):
    """Preprocess the IMUPoser dataset"""

    train_split = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
    test_split = ['P9', 'P10']
    subjects = train_split if split == "train" else test_split

    accs, oris, poses, trans = [], [], [], []
    for pid_path in sorted(paths.raw_imuposer.iterdir()):
        if pid_path.name not in subjects:
            continue

        print(f"Processing: {pid_path.name}")
        for fpath in sorted(pid_path.iterdir()):
            with open(fpath, "rb") as f: 
                fdata = pickle.load(f)
                
                acc = fdata['imu'][:, :5*3].view(-1, 5, 3)
                ori = fdata['imu'][:, 5*3:].view(-1, 5, 3, 3)
                pose = math.axis_angle_to_rotation_matrix(fdata['pose']).view(-1, 24, 3, 3)
                tran = fdata['trans'].to(torch.float32)
                
                 # align IMUPoser global fame with DIP
                rot = torch.tensor([[[-1, 0, 0], [0, 0, 1], [0, 1, 0.]]])
                pose[:, 0] = rot.matmul(pose[:, 0])
                tran = tran.matmul(rot.squeeze())

                # ensure sizes are consistent
                assert tran.shape[0] == pose.shape[0]

                accs.append(acc)    # N, 5, 3
                oris.append(ori)    # N, 5, 3, 3
                poses.append(pose)  # N, 24, 3, 3
                trans.append(tran)  # N, 3

    print(f"# Data Processed: {len(accs)}")
    data = {
        'acc': accs,
        'ori': oris,
        'pose': poses,
        'tran': trans
    }
    data_path = paths.eval_dir / f"imuposer_{split}.pt"
    torch.save(data, data_path)


def process_nymeria(split="train"):
    """Process Nymeria dataset for training and evaluation.
    
    This function processes the Nymeria dataset, which contains XSens motion capture data,
    and converts it into the format required by MobilePoser. The processing steps include:
    1. Loading XSens motion data from .npz files
    2. Converting quaternions to axis-angle representation
    3. Downsampling from 240Hz to the target FPS
    4. Aligning coordinate systems between Nymeria and SMPL
    5. Synthesizing IMU accelerations and orientations
    6. Computing foot-ground contact probabilities
    7. Saving processed data in .pt format for training or evaluation
    
    Args:
        split (str): Dataset split to process, either "train" or "test"
    """
    def _foot_ground_probs(joint):
        """Compute foot-ground contact probabilities based on foot movement.
        
        This function detects when feet are in contact with the ground by measuring
        the distance between consecutive foot positions. Small movements indicate
        the foot is likely in contact with the ground.
        
        Args:
            joint: Tensor of joint positions
            
        Returns:
            Tensor of foot contact probabilities (left and right feet)
        """
        dist_lfeet = torch.norm(joint[1:, 10] - joint[:-1, 10], dim=1)
        dist_rfeet = torch.norm(joint[1:, 11] - joint[:-1, 11], dim=1)
        lfoot_contact = (dist_lfeet < 0.008).int()  # Threshold for contact detection
        rfoot_contact = (dist_rfeet < 0.008).int()
        lfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), lfoot_contact))
        rfoot_contact = torch.cat((torch.zeros(1, dtype=torch.int), rfoot_contact))
        return torch.stack((lfoot_contact, rfoot_contact), dim=1)

    try:
        processed = [fpath.name for fpath in (paths.processed_datasets).iterdir()]
    except FileNotFoundError:
        processed = []

    test_split = []  # Add test sequences here
    train_split = datasets.nymeria_datasets
    sequences = train_split if split == "train" else test_split
    
    if f"nymeria_{split}.pt" in processed:
        print(f"Nymeria {split} dataset already processed. Skipping.")
        return

    data_pose, data_trans, data_beta, length = [], [], [], []
    print(f"\rReading Nymeria {split} dataset")

    # Process each sequence in the dataset
    for seq_name in tqdm(sequences):
        try:
            seq_path = os.path.join(paths.raw_nymeria, seq_name)
            
            xsens_npz_paths = [
                os.path.join(seq_path, "body", "xdata.npz"),
                os.path.join(seq_path, "xsens", "xdata.npz"),
                os.path.join(seq_path, "xdata.npz")
            ]
            
            xsens_npz_path = None
            for path in xsens_npz_paths:
                if os.path.exists(path):
                    xsens_npz_path = path
                    break
            
            # Skip if no XSens data is found
            if xsens_npz_path is None:
                print(f"XSens data not found for {seq_name}, skipping")
                continue
                
            xsens_data = np.load(xsens_npz_path)
            
            if XSensConstants.k_part_qWXYZ not in xsens_data or XSensConstants.k_part_tXYZ not in xsens_data:
                print(f"Required XSens data keys not found in {seq_name}, skipping")
                print(f"Available keys: {list(xsens_data.keys())}")
                continue
            
            q_wxyz = xsens_data[XSensConstants.k_part_qWXYZ].reshape(-1, XSensConstants.num_parts, 4)
            t_xyz = xsens_data[XSensConstants.k_part_tXYZ].reshape(-1, XSensConstants.num_parts, 3)
            
            step = max(1, round(240 / TARGET_FPS))  # Nymeria is recorded at 240Hz
            
            for i in range(0, len(q_wxyz), step):
                quat = torch.tensor(q_wxyz[i])
                trans = torch.tensor(t_xyz[i])
                
                pose = math.quaternion_to_axis_angle(quat)
                
                data_pose.append(pose.reshape(-1).numpy())
                data_trans.append(trans[0].numpy())  # Use pelvis as root joint
                data_beta.append(np.zeros(10))  # Use default SMPL shape parameters
                
            # Store the length of the sequence after downsampling
            length.append(len(range(0, len(q_wxyz), step)))
            
        except Exception as e:
            print(f"Error processing sequence {seq_name}: {e}")
            continue

    if len(data_pose) == 0:
        print(f"No valid sequences found in Nymeria dataset")
        return

    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, XSensConstants.num_parts*3)
    
    pose_smpl = torch.zeros((pose.shape[0], 24, 3), dtype=torch.float32)
    for i in range(min(XSensConstants.num_parts, 24)):
        pose_smpl[:, i] = pose[:, i*3:(i+1)*3]
    
    nymeria_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    
    # Transform translations
    tran = nymeria_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    
    pose_smpl[:, 0] = math.rotation_matrix_to_axis_angle(
        nymeria_rot.matmul(math.axis_angle_to_rotation_matrix(pose_smpl[:, 0])))

    print("Synthesizing IMU accelerations and orientations")
    b = 0  # Base index for the current sequence
    
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, out_contact = [], [], [], [], [], [], []
    
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: 
            b += l
            print("\tdiscard one sequence with length", l)
            continue
            
        p = math.axis_angle_to_rotation_matrix(pose_smpl[b:b + l]).view(-1, 24, 3, 3)
        
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

        out_pose.append(p.clone())                           # Pose as rotation matrices (N, 24, 3, 3)
        out_tran.append(tran[b:b + l].clone())               # Global translation (N, 3)
        out_shape.append(shape[i].clone())                   # SMPL shape parameters (10)
        out_joint.append(joint[:, :24].contiguous().clone()) # Joint positions (N, 24, 3)
        out_vacc.append(_syn_acc(vert[:, vi_mask]))          # Synthetic accelerations (N, 6, 3)
        out_contact.append(_foot_ground_probs(joint).clone()) # Foot contact (N, 2)

        out_vrot.append(grot[:, ji_mask])                    # Global rotations (N, 6, 3, 3)
        b += l

    print("Saving processed data...")
    data = {
        'joint': out_joint,      # Joint positions
        'pose': out_pose,        # Pose as rotation matrices
        'shape': out_shape,      # SMPL shape parameters
        'tran': out_tran,        # Global translations
        'acc': out_vacc,         # Synthetic accelerations
        'ori': out_vrot,         # Global rotations for IMU locations
        'contact': out_contact   # Foot-ground contact probabilities
    }
    
    data_path = paths.eval_dir / f"nymeria_{split}.pt" if split == "test" else paths.processed_datasets / f"nymeria_{split}.pt"
    torch.save(data, data_path)
    print(f"Processed Nymeria {split} dataset is saved at: {data_path}")


def create_directories():
    paths.processed_datasets.mkdir(exist_ok=True, parents=True)
    paths.eval_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="amass")
    args = parser.parse_args()

    # create dataset directories
    create_directories()

    # process datasets
    if args.dataset == "amass":
        process_amass()
    elif args.dataset == "totalcapture":
        process_totalcapture()
    elif args.dataset == "imuposer":
        process_imuposer(split="train")
        process_imuposer(split="test")
    elif args.dataset == "dip":
        process_dipimu(split="train")
        process_dipimu(split="test")
    elif args.dataset == "nymeria":
        process_nymeria(split="train")
        process_nymeria(split="test")
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

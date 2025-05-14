import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import glob
import sys
import signal
from pathlib import Path

sys.path.append('/home/naoto/docker_workspace/nymeria_dataset')
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

    test_split = ['s_03']
    train_split = ['s_01', 's_02']
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

    train_split = ['P1']
    test_split = ['P2']
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


def process_nymeria(resume: bool = True, contact_logic: str = "xdata", max_sequences: int = -1, imu_device: str = "aria"):
    """
    Preprocess the Nymeria dataset and export it in the same AMASS-compatible
    structure that the other *process_* utilities in this file produce.

    Parameters
    ----------
    resume : bool, default True
        When True the function keeps appending to the previously written
        nymeria_train.pt / nymeria_test.pt files so that the preprocessing can
        be resumed.  When False those files are deleted and the whole dataset
        is regenerated.
    contact_logic : {"xdata", "amass"}, default "xdata"
        "xdata" – use the `foot_contacts` attribute stored in the XSens npz.
        "amass" – ignore that attribute and infer foot contact with the AMASS
        heuristic (frame-to-frame foot displacement < 0.8 cm).
    max_sequences : int, default -1
        Number of sequences to process. If < 0, process all sequences.
    imu_device : {"aria", "xsens"}, default "aria"
        "aria" – use IMU data from Aria device (head, rwrist, lwrist).
        "xsens" – use IMU data from XSens (left wrist, right wrist, left thigh, right thigh, head, pelvis).
    """
    # ------------------------------------------------------------------
    # Imports (local to avoid slowing down CLI start-up for other datasets)
    # ------------------------------------------------------------------
    import os, glob, numpy as np, torch
    from pathlib import Path
    from tqdm import tqdm

    from nymeria.data_provider import NymeriaDataProvider
    from nymeria.xsens_constants import XSensConstants
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    from mobileposer.articulate.model import ParametricModel
    from mobileposer.articulate.math.angular import quaternion_to_rotation_matrix
    from mobileposer.config import paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    TARGET_FPS = 30  # keep consistent with the other datasets

    def _foot_ground_probs(joint_sequence: torch.Tensor) -> torch.Tensor:
        """AMASS-style foot contact detection (distance threshold)."""
        dist_l = torch.norm(joint_sequence[1:, 10] - joint_sequence[:-1, 10], dim=1)
        dist_r = torch.norm(joint_sequence[1:, 11] - joint_sequence[:-1, 11], dim=1)
        l_fc = (dist_l < 0.008).int()
        r_fc = (dist_r < 0.008).int()
        # prepend one zero so that length matches T
        l_fc = torch.cat((torch.zeros(1, dtype=torch.int32), l_fc))
        r_fc = torch.cat((torch.zeros(1, dtype=torch.int32), r_fc))
        return torch.stack((l_fc, r_fc), dim=1)  # (T, 2)

    # Mapping: XSens segment name –> SMPL joint index
    seg2smpl = {
        "Pelvis": 0,
        "L5": 3,
        "L3": 6,
        "T12": 9,
        "T8": 12,
        "Neck": 12,  # same as T8/neck for safety
        "Head": 15,
        "R_Shoulder": 14,
        "R_UpperArm": 17,
        "R_Forearm": 19,
        "R_Hand": 21,
        "L_Shoulder": 13,
        "L_UpperArm": 16,
        "L_Forearm": 18,
        "L_Hand": 20,
        "R_UpperLeg": 2,
        "R_LowerLeg": 5,
        "R_Foot": 8,
        "R_Toe": 11,
        "L_UpperLeg": 1,
        "L_LowerLeg": 4,
        "L_Foot": 7,
        "L_Toe": 10,
    }
    segname2idx = {n: i for i, n in enumerate(XSensConstants.part_names)}
    smpl_to_seg = torch.full((24,), -1, dtype=torch.long)
    for seg_name, smpl_idx in seg2smpl.items():
        if seg_name in segname2idx:
            smpl_to_seg[smpl_idx] = segname2idx[seg_name]

    # ------------------------------------------------------------------
    # Prepare output containers & resume if requested
    # ------------------------------------------------------------------
    # ファイル名にIMUデバイスとcontact_logicを含める
    device_suffix = f"_{imu_device}"
    contact_suffix = f"_{contact_logic}"
    train_path = paths.processed_datasets / f"nymeria{device_suffix}{contact_suffix}_train.pt"
    test_path = paths.eval_dir / f"nymeria{device_suffix}{contact_suffix}_test.pt"

    def _empty_container():
        return {k: [] for k in [
            "joint", "pose", "shape", "tran", "acc", "ori", "contact"]}

    train_data = _empty_container()
    test_data = _empty_container()
    processed_idxs: set[int] = set()

    def _remove_last_sequence(data_container):
        """最後のシーケンスを削除します。エラー終了した場合に不完全なデータが残る可能性があるため。"""
        if not data_container.get("joint", []):
            return 0  # 空のコンテナ
            
        sequence_count = len(data_container["joint"])
        if sequence_count > 0:
            # 最後のシーケンスを削除
            for k in data_container:
                if len(data_container[k]) > 0:  # 安全のためチェック
                    data_container[k] = data_container[k][:-1]
            return len(data_container["joint"])
        return 0
            
    if resume:
        num_train_sequences = 0
        num_test_sequences = 0
        
        if train_path.exists():
            try:
                buf = torch.load(train_path)
                for k in train_data:
                    train_data[k] = buf.get(k, [])
                
                # 最後のシーケンスを削除（エラー終了の可能性あり）
                if len(train_data["joint"]) > 0:
                    old_count = len(train_data["joint"])
                    num_train_sequences = _remove_last_sequence(train_data)
                    print(f"[Safety] 訓練データから最後のシーケンスを削除しました。{old_count} → {num_train_sequences}")
                else:
                    num_train_sequences = 0
                
                # トレーニングインデックスを追加 (0 ~ num_train-1)
                processed_idxs.update(range(num_train_sequences))
                print(f"[Resume] loaded {num_train_sequences} training sequences.")
            except Exception as e:
                print(f"[Warning] 訓練データ読み込み中にエラー: {e}")
                print("訓練データファイルをリセットします")
                train_path.unlink(missing_ok=True)
                train_data = _empty_container()
            
        if test_path.exists():
            try:
                buf = torch.load(test_path)
                for k in test_data:
                    test_data[k] = buf.get(k, [])
                
                # 最後のシーケンスを削除（エラー終了の可能性あり）
                if len(test_data["joint"]) > 0:
                    old_count = len(test_data["joint"])
                    num_test_sequences = _remove_last_sequence(test_data)
                    print(f"[Safety] テストデータから最後のシーケンスを削除しました。{old_count} → {num_test_sequences}")
                else:
                    num_test_sequences = 0
                
                # テストインデックスを追加 (各10個に1つ目のインデックス: 9, 19, 29, ...)
                for i in range(num_test_sequences):
                    # 実際に処理済みのテストインデックスを特定するのは難しいため
                    # 安全のため、すべての可能性のあるテストインデックスを追加
                    for test_idx in range(9, len(seq_dirs), 10):
                        processed_idxs.add(test_idx)
                print(f"[Resume] loaded {num_test_sequences} testing sequences.")
            except Exception as e:
                print(f"[Warning] テストデータ読み込み中にエラー: {e}")
                print("テストデータファイルをリセットします")
                test_path.unlink(missing_ok=True)
                test_data = _empty_container()
            
        print(f"[Resume] total processed sequences: {len(processed_idxs)}")
    else:
        train_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Discover Nymeria sequence directories early
    # ------------------------------------------------------------------
    # `seq_dirs` is needed during the resume logic below to determine which
    # indices have already been processed. Defining it here ensures it is
    # available and prevents an UnboundLocalError that occurred when the
    # variable was referenced before assignment.
    seq_dirs = sorted(glob.glob(os.path.join(str(paths.raw_nymeria), "*")))

    # ------------------------------------------------------------------
    # Iterate over recordings ------------------------------------------------
    # ------------------------------------------------------------------
    smpl_model = ParametricModel(paths.smpl_file)  # one global instance
    parent = smpl_model.parent

    # 10シーケンスに1つをテストデータとする（seq_idx % 10 == 9 のものがテスト）

    def _save():
        os.makedirs(paths.processed_datasets, exist_ok=True)
        os.makedirs(paths.eval_dir, exist_ok=True)
        torch.save(train_data, train_path)
        torch.save(test_data, test_path)
        print("[AutoSave] progress written to disk.")

    processed_count = 0
    try:
        def signal_handler(sig, frame):
            print(f"\n[Signal {sig}] Saving progress before termination...")
            _save()
            sys.exit(0)
        
        # 各種終了シグナルに対するハンドラを登録
        signal.signal(signal.SIGTERM, signal_handler)  # kill コマンドのデフォルト
        signal.signal(signal.SIGINT, signal_handler)   # キーボード割り込み (Ctrl+C)
        # SIGKILL (kill -9) は捕捉できないため登録しない

        for seq_idx, seq_dir in enumerate(tqdm(seq_dirs, desc="Nymeria")):
            if seq_idx in processed_idxs:
                continue
            if max_sequences > 0 and processed_count >= max_sequences:
                print(f"Reached max_sequences ({max_sequences}). Stopping early.")
                break

            # --------------------------------------------------------------
            # Load Nymeria sequence
            # --------------------------------------------------------------
            try:
                dp = NymeriaDataProvider(sequence_rootdir=Path(seq_dir))
                body_dp = dp.body_dp
                head_p, lw_p, rw_p = dp.recording_head, dp.recording_lwrist, dp.recording_rwrist
            except RuntimeError as e:
                print(f"[Warning] Skipping sequence due to error in NymeriaDataProvider: {seq_dir}\n  {e}")
                continue

            # XSens arrays
            xs_q = body_dp.xsens_data[XSensConstants.k_part_qWXYZ].reshape(-1, XSensConstants.num_parts, 4)
            xs_t = body_dp.xsens_data[XSensConstants.k_part_tXYZ].reshape(-1, XSensConstants.num_parts, 3)
            xs_ts = body_dp.xsens_data[XSensConstants.k_timestamps_us]  # μs
            xs_fc = body_dp.xsens_data.get(XSensConstants.k_foot_contacts, None)

            # Build timestamp query grid (ns)
            t_start, t_end = dp.timespan_ns
            step_ns = int(1e9 / TARGET_FPS)
            query_ns = np.arange(t_start, t_end, step_ns, dtype=np.int64)

            # Per-frame containers
            joint_seq, pose_seq, tran_seq = [], [], []
            contact_seq: list[torch.Tensor] = []  # (2,) int each
            ori_frames: list[np.ndarray] = []  # (6?,4)
            acc_frames: list[np.ndarray] = []
            betas = torch.zeros(10)

            # Helper to fetch IMU samples (unchanged from original implementation)
            def _imu_at(provider, stream_id: str, t_ns: int):
                if provider is None or provider.vrs_dp is None:
                    return np.zeros(4), np.zeros(3)  # NaN ではなく 0 を返す
                try:
                    imu, _ = provider.vrs_dp.get_imu_data_by_time_ns(
                        StreamId(stream_id), time_ns=int(t_ns),
                        time_domain=TimeDomain.TIME_CODE,
                        time_query_options=TimeQueryOptions.CLOSEST,
                    )
                    q = np.array([imu.w, imu.x, imu.y, imu.z], dtype=np.float32)
                    acc = np.array([imu.ax, imu.ay, imu.az], dtype=np.float32)
                    return q, acc
                except Exception:
                    return np.zeros(4), np.zeros(3)  # NaN ではなく 0 を返す

            for t_ns in query_ns:
                # ------------------------------------------------------
                # XSens frame selection (closest)
                # ------------------------------------------------------
                t_us = int(t_ns // 1000)
                ridx = np.searchsorted(xs_ts, t_us)
                if ridx == len(xs_ts):
                    ridx -= 1
                if ridx > 0 and abs(xs_ts[ridx] - t_us) > abs(xs_ts[ridx - 1] - t_us):
                    ridx -= 1

                part_q = xs_q[ridx]  # (23,4)
                part_t = xs_t[ridx]  # (23,3)

                # Global rotations for the 24 SMPL joints -----------------
                R_global = torch.eye(3).repeat(24, 1, 1)
                for j in range(24):
                    seg_index = smpl_to_seg[j].item()
                    if seg_index >= 0:
                        R_global[j] = quaternion_to_rotation_matrix(
                            torch.from_numpy(part_q[seg_index]).float().unsqueeze(0)
                        )[0]

                # Convert to local rotations
                R_local = smpl_model.inverse_kinematics_R(R_global.unsqueeze(0))[0]
                pose_seq.append(R_local.unsqueeze(0))

                # Joint positions & root translation ---------------------
                joint_pos = torch.zeros(24, 3)
                for j in range(24):
                    seg_index = smpl_to_seg[j].item()
                    if seg_index >= 0:
                        joint_pos[j] = torch.from_numpy(part_t[seg_index]).float()
                joint_seq.append(joint_pos)
                tran_seq.append(joint_pos[0].unsqueeze(0))

                # Foot contact -------------------------------------------
                if contact_logic == "xdata" and xs_fc is not None:
                    contact_seq.append(torch.from_numpy(xs_fc[ridx]).int())
                else:
                    # placeholder; computed later
                    contact_seq.append(None)

                # IMU ----------------------------------------------------
                if imu_device == "aria":
                    # -------------------------
                    # Aria sensors (4 IMUs)
                    # -------------------------
                    qhL, ahL = _imu_at(head_p, "1202-2", t_ns)  # Head left (unused)
                    qhR, ahR = _imu_at(head_p, "1202-1", t_ns)  # Head right (used)
                    qlw, alw = _imu_at(lw_p, "1202-2", t_ns)   # Left wrist
                    qrw, arw = _imu_at(rw_p, "1202-1", t_ns)   # Right wrist

                    # Create XSens-like container (23 segments)
                    num_parts = XSensConstants.num_parts  # 23
                    ori_part = np.zeros((num_parts, 4), dtype=np.float32)
                    acc_part = np.zeros((num_parts, 3), dtype=np.float32)

                    # Fill Aria-provided sensors
                    ori_part[1] = qhR; acc_part[1] = ahR     # head (use right-side IMU)
                    ori_part[2] = qlw;  acc_part[2] = alw    # left wrist
                    ori_part[3] = qrw;  acc_part[3] = arw    # right wrist

                    # -------------------------
                    # Supplement with XSens data
                    # -------------------------
                    xsens_mapping = {0: 0, 4: 4, 7: 7}  # pelvis, left thigh, right thigh

                    # Orientation from XSens quaternion array
                    for seg in xsens_mapping.values():
                        ori_part[seg] = part_q[seg]

                    # Compute acceleration for the required segments
                    # Determine neighboring XSens indices for finite-difference
                    if len(ori_frames) > 0 and len(ori_frames) < len(query_ns) - 1:
                        prev_idx = np.searchsorted(xs_ts, int(query_ns[len(ori_frames) - 1] // 1000))
                        next_idx = np.searchsorted(xs_ts, int(query_ns[len(ori_frames) + 1] // 1000))
                    else:
                        prev_idx = next_idx = None

                    for seg in xsens_mapping.values():
                        if prev_idx is not None and next_idx is not None and prev_idx < len(xs_t) and next_idx < len(xs_t):
                            prev_pos = xs_t[prev_idx][seg]
                            curr_pos = xs_t[ridx][seg]
                            next_pos = xs_t[next_idx][seg]
                            acc_part[seg] = (next_pos + prev_pos - 2 * curr_pos) * (TARGET_FPS ** 2)

                    # Store one frame
                    ori_frames.append(ori_part)
                    acc_frames.append(acc_part)

                elif imu_device == "xsens":
                    # XSensデータからIMU情報を抽出
                    num_parts: int = XSensConstants.num_parts  # = 23

                    # Orientation : 既に (23, 4) 形式で取得可能
                    ori_part = xs_q[ridx].astype(np.float32)  # (23, 4)

                    # Acceleration : XSensデータからIMU情報を抽出
                    acc_part = np.zeros((num_parts, 3), dtype=np.float32)

                    # AMASSで使用する6箇所のセグメントID
                    xsens_mapping = {
                        0: 15,  # left wrist
                        1: 18,  # right wrist
                        2: 4,   # left thigh
                        3: 7,   # right thigh
                        4: 19,  # head
                        5: 0,   # pelvis
                    }

                    # 加速度推定に必要な前後フレームのインデックスを取得
                    if len(ori_frames) > 0 and len(ori_frames) < len(query_ns) - 1:
                        prev_idx = np.searchsorted(xs_ts, int(query_ns[len(ori_frames) - 1] // 1000))
                        next_idx = np.searchsorted(xs_ts, int(query_ns[len(ori_frames) + 1] // 1000))
                    else:
                        prev_idx = next_idx = None

                    for amass_idx, xsens_idx in xsens_mapping.items():
                        # 加速度を推定 (位置データが利用可能な場合)
                        if prev_idx is not None and next_idx is not None and prev_idx < len(xs_t) and next_idx < len(xs_t):
                            prev_pos = xs_t[prev_idx][xsens_idx]
                            curr_pos = xs_t[ridx][xsens_idx]
                            next_pos = xs_t[next_idx][xsens_idx]
                            acc_part[xsens_idx] = (next_pos + prev_pos - 2 * curr_pos) * (TARGET_FPS ** 2)

                    # 1フレーム分を保存
                    ori_frames.append(ori_part)
                    acc_frames.append(acc_part)

            # --------------------------------------------------------------
            # Finalise per-sequence tensors
            # --------------------------------------------------------------
            joints_tensor = torch.stack(joint_seq)  # (T,24,3)
            poses_tensor = torch.cat(pose_seq, dim=0)  # (T,24,3,3)
            trans_tensor = torch.cat(tran_seq, dim=0)  # (T,3)

            if contact_logic == "amass":
                contacts_tensor = _foot_ground_probs(joints_tensor)
            else:
                contacts_tensor = torch.stack([c if c is not None else torch.zeros(2, dtype=torch.int32)
                                               for c in contact_seq])

            # Map Nymeria-available IMUs (head_r / wrist etc.) into AMASS order
            T = len(ori_frames)
            acc_np = np.array(acc_frames, dtype=np.float32)  # (T,23,3)
            ori_np = np.array(ori_frames, dtype=np.float32)  # (T,23,4)

            # NaN値のチェックと修正
            acc_np = np.nan_to_num(acc_np, nan=0.0)
            ori_np = np.nan_to_num(ori_np, nan=0.0)

            imu_num = 6
            acc_amass = np.zeros((T, imu_num, 3), dtype=np.float32)
            ori_amass = np.zeros((T, imu_num, 4), dtype=np.float32)
            # order mapping: left wrist, right wrist, left thigh, right thigh, head, pelvis
            if imu_device == "aria":
                # Set Aria sensor data for available positions
                acc_amass[:, 1] = acc_np[:, 3]  # right wrist
                acc_amass[:, 0] = acc_np[:, 2]  # left wrist
                acc_amass[:, 4] = acc_np[:, 1]  # head (use right head IMU)
                ori_amass[:, 1] = ori_np[:, 3]
                ori_amass[:, 0] = ori_np[:, 2]
                ori_amass[:, 4] = ori_np[:, 1]
                
                # Supplement missing positions (left thigh, right thigh, pelvis) with XSens data already embedded in acc_np / ori_np
                xsens_mapping = {
                    2: 4,   # left thigh
                    3: 7,   # right thigh
                    5: 0,   # pelvis
                }

                for amass_idx, xsens_idx in xsens_mapping.items():
                    acc_amass[:, amass_idx] = acc_np[:, xsens_idx]
                    ori_amass[:, amass_idx] = ori_np[:, xsens_idx]
                
            elif imu_device == "xsens":
                # XSensデータからIMU情報を抽出
                # order mapping: left wrist, right wrist, left thigh, right thigh, head, pelvis
                # XSensの部位IDマップ（例: left_wrist -> 15, right_wrist -> 18）
                xsens_mapping = {
                    0: 15,  # left wrist
                    1: 18,  # right wrist
                    2: 4,   # left thigh
                    3: 7,   # right thigh
                    4: 19,  # head
                    5: 0,   # pelvis
                }

                for amass_idx, xsens_idx in xsens_mapping.items():
                    acc_amass[:, amass_idx] = acc_np[:, xsens_idx]
                    ori_amass[:, amass_idx] = ori_np[:, xsens_idx]

            sample = {
                "joint": joints_tensor,
                "pose": poses_tensor,
                "shape": betas,
                "tran": trans_tensor,
                "acc": torch.from_numpy(acc_amass),
                "ori": torch.from_numpy(ori_amass),
                "contact": contacts_tensor,
            }

            # データにNaNがないかチェック
            has_nan = False
            for k, v in sample.items():
                if torch.isnan(v).any():
                    print(f"Warning: NaN detected in '{k}' for sequence {seq_idx}")
                    has_nan = True
                    break

            if has_nan:
                print(f"Skipping sequence {seq_idx} due to NaN values")
                continue

            # 10シーケンスに1つをテストデータに回す
            if seq_idx % 10 == 9:  # 0, 1, 2, ..., 8がトレーニング、9がテスト
                for k in test_data:
                    test_data[k].append(sample[k])
                print(f"[Test] Adding sequence {seq_idx} to test dataset")
            else:
                for k in train_data:
                    train_data[k].append(sample[k])
                print(f"[Train] Adding sequence {seq_idx} to train dataset")

            processed_idxs.add(seq_idx)
            processed_count += 1
            _save()

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving progress…")
        _save()
        return

    _save()
    print(f"Finished Nymeria preprocessing → {train_path} & {test_path}")


def create_directories():
    paths.processed_datasets.mkdir(exist_ok=True, parents=True)
    paths.eval_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--restart", action="store_true", help="Start preprocessing from scratch (do not resume)")
    parser.add_argument("--max-sequences", type=int, default=-1, help="Number of sequences to process (default: all)")
    parser.add_argument("--contact-logic", type=str, default="xdata", help="Contact logic for Nymeria")
    parser.add_argument("--imu-device", type=str, default="aria", choices=["aria", "xsens"], help="IMU device to use for Nymeria")
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
        process_nymeria(resume=not args.restart, max_sequences=args.max_sequences, contact_logic=args.contact_logic, imu_device=args.imu_device)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

# MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices
Author's implementation of the paper [MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices](https://dl.acm.org/doi/pdf/10.1145/3654777.3676461). This work was published at UIST'24.

<br>
<div align="center">
<img src="teaser.gif" alt="teaser.gif" width="100%">
</div>
<br>

## Installation 
We recommend configuring the project inside an Anaconda environment. We have tested everything using [Anaconda](https://docs.anaconda.com/anaconda/install/) version 23.9.0 and Python 3.9. The first step is to create a virtual environment, as shown below (named `mobileposer310`).
```
conda create -n mobileposer310 python=3.10
```
You should then activate the environment as shown below. All following operations must be completed within the virtual environment.
```
(conda init)
(source ~/.bashrc)
conda activate mobileposer310
```
Then, install the required packages.
```
pip install -r requirements.txt
```
You will then need to install the local mobileposer package for development via the command below. You must run this from the root directory (e.g., where setup.py is).
```
pip install -e .
```

## Process Datasets

### Download Training Data
1. Register and download the AMASS dataset from [here](https://amass.is.tue.mpg.de/). We use 'SMPLH+G' for each dataset. 
2. Register and download the DIP-IMU dataset from [here](https://dip.is.tuebingen.mpg.de/). Download the raw (unormalized) data.
3. Request access to the TotalCapture dataset [here](https://cvssp.org/data/totalcapture/). Download Vicon Groundtruth in the raw folder, and IMU data in the IMU folder. 
4. Download the IMUPoser dataset from [here](https://github.com/FIGLAB/IMUPoser).
5. The Nymeria dataset can be accessed from [here](https://www.projectaria.com/datasets/nymeria/). The dataset is structured as described in the [nawta/nymeria_dataset](https://github.com/nawta/nymeria_dataset) repository.

Once downloaded, your directory might appear as follows:
```bash
data
└── raw
    ├── AMASS
    │   ├── ACCAD
    │   ├── BioMotionLab_NTroje
    │   ├── BMLhandball
    │   ├── ...
    │   └── Transitions_mocap
    ├── DIP_IMU
    │   ├── s_01
    │   ├── s_02
    │   ├── s_03
    │   ├── ...
    │   └── s_10
    ├── IMUPoser
    │   ├── P1
    │   ├── P2
    │   ├── P3
    │   ├── ...
    │   └── P10
    └── TotalCapture/
            ├── IMU/
            │   ├── s1_acting1.pkl
            │   ├── ...
            └── raw/
                ├── S1/
                │   ├── acting1/
                │   │   ├── gt_skel_gbl_ori.txt
                │   │   ├── gt_skel_gbl_pos.txt
                │   ├── ...
```

### Setup Training Data 
In `config.py`: 
- Set `paths.processed_datasets` to the directory containing the pre-processed datasets.
- Set `paths.raw_amass` to the directory containing the AMASS dataset.
- Set `paths.raw_dip` to the directory containing the DIP dataset.
- Set `paths.raw_imuposer` to the directory containing the IMUPoser dataset.
- Set `paths.raw_nymeria` to the directory containing the Nymeria dataset.
  
The script `process.py` drives the dataset pre-processing. This script takes the following parameters:
1. `--dataset`: Dataset to pre-process (`amass`, `dip`, `imuposer`, `nymeria`). Defaults to `amass`.
2. `--restart`: Start preprocessing from scratch (do not resume from previous progress).
3. `--max_sequences`: Number of sequences to process (default: all sequences). If not specified or set to -1, all sequences will be processed. This is useful for debugging or partial dataset generation.
4. `--imu-device`: IMU device to use for Nymeria dataset processing (`aria` or `xsens`). Defaults to `aria`. Only applicable when `--dataset nymeria` is selected.
5. `--contact-logic`: Contact logic to use for Nymeria dataset processing (e.g., `xdata`, `legacy`). Defaults to `xdata`. Only applicable when `--dataset nymeria` is selected. `legacy` is the contact logic used in the original implementation of preprocess AMASS dataset.

As an example, the following commands will pre-process the DIP and Nymeria datasets:
```
$ PYTHONPATH=. python mobileposer/process.py --dataset nymeria
$ PYTHONPATH=. python mobileposer/process.py --dataset dip
$ PYTHONPATH=. python mobileposer/process.py --dataset nymeria --imu-device aria --contact-logic xdata
$ PYTHONPATH=. python mobileposer/process.py --dataset nymeria --imu-device xsens --contact-logic legacy
$ PYTHONPATH=. python mobileposer/process.py --dataset nymeria --max_sequences 10 --imu-device xsens  # Only process 10 sequences (for debug)
```

#### Nymeria IMU Device Selection & Output Naming
When processing the Nymeria dataset, you can select the IMU data source using the `--imu-device` option:
- `aria`: Uses head, right wrist, and left wrist IMU values (default).
- `xsens`: Uses six IMU values (left wrist, right wrist, left thigh, right thigh, head, pelvis).

The output files generated will include the selected IMU device and contact logic in their filenames, e.g., `nymeria_xsens_xdata_train.pt`.

All missing IMU values are handled robustly: any NaN values are replaced with 0 during processing.

## Training Models 
The script `train.py` drives the training process. This script takes the following parameters:
1. `--module`: Train an individual module (`poser`, `joints`, `foot_contact`, `velocity`). Default to training all modules. 
2. `--init-from`: Initialize training from an existing checkpoint. Defaults to training from scratch. 
3. `--finetune`: Specify dataset for finetuning module (e.g., `dip`). 
4. `--fast-dev-run`: A boolean flag that caps the execution to a single epoch. This flag is useful for debugging.

As an example, we can execute the following command to train all modules: 
```
$ python train.py
```

### Streaming Training Mode
For large datasets or memory-constrained environments, use the streaming training mode:
```
$ python train.py --stream
```

The streaming mode loads data on-the-fly to reduce memory usage. This is particularly useful for:
- Systems with limited RAM
- Very large datasets that don't fit in memory
- Continuous training on new data

#### Memory Usage Configuration
Memory usage can be controlled through several parameters in `mobileposer/config.py`:

**Key Memory Control Parameters:**

1. **`stream_buffer_size`** - Controls how many sequences are buffered in memory during streaming
   ```python
   # For systems with limited memory (< 32GB RAM)
   stream_buffer_size = 500
   
   # For systems with moderate memory (32-64GB RAM)  
   stream_buffer_size = 1000  # Current stable setting
   
   # For systems with high memory (> 64GB RAM)
   stream_buffer_size = 5000-20000
   ```

2. **`batch_size`** - Batch size is automatically adjusted for streaming mode
   ```python
   # Original batch size (non-streaming)
   batch_size = 4096
   
   # Automatically reduced for streaming (in data.py)
   # Large batches: max(512, original_bs // 8) 
   # Small batches: max(32, original_bs // 2)
   ```

3. **`accumulate_grad_batches`** - Maintains effective batch size through gradient accumulation
   ```python
   # Balances memory usage vs training quality
   accumulate_grad_batches = 16  # Current stable setting
   ```

4. **`num_workers`** - Controls parallel data loading workers
   ```python
   # Automatically adjusted for streaming mode
   num_workers = 8  # Reduced from 32 for streaming
   ```

**Memory Usage Guidelines:**

| System Memory | Recommended Settings | Expected Usage |
|---------------|---------------------|----------------|
| 16-32 GB | `stream_buffer_size=500`, `batch_size=256` | 8-15 GB |
| 32-64 GB | `stream_buffer_size=1000`, `batch_size=512` | 15-30 GB |
| 64-128 GB | `stream_buffer_size=2000-5000`, `batch_size=512-1024` | 30-60 GB |
| > 128 GB | `stream_buffer_size=10000+`, `batch_size=1024` | 60+ GB |

**GPU Memory Considerations:**
- RTX 4090 (24GB): Current settings use ~4.3GB (18% utilization)
- RTX 3080 (10GB): Use `batch_size=256` or lower
- RTX 3060 (8GB): Use `batch_size=128` and reduce `stream_buffer_size=500`

**Monitoring Memory Usage:**
```bash
# Monitor system memory
watch -n 1 free -h

# Monitor GPU memory  
watch -n 1 nvidia-smi

# Run with memory monitoring
python monitor_training.py  # Custom monitoring script
```

**Environment Variables for Memory Optimization:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0
```

### Finetuning Model
To facilitate finetuning MobilePoser, we provide a script `finetune.sh`. To run this script, use the following syntax: 
```
$ ./finetune.sh <dataset-name> <checkpoint-directory>
```

### Prepare Model
The script `combine_weights.py` combines the weights of individual modules into a single weight file that can be loaded into `MobilePoserNet`. 
To run this script, use the following syntax: 
```
$ python combine_weights.py --finetune <dataset-name> --checkpoint <checkpoint-directory>
```
Omit the `--finetune` argument if you did not finetune. The resulting weight file will be stored under the same directory as the `checkpoint-directory`

### Download pre-trained network weights
We provide a pre-trained model for the set of configurations listed in `config.py`. 
1. Download weights from [here](https://uchicago.box.com/s/ey3y49srpo79propzvmjx0t8u3ael6cl). 
2. In `config.py`, set the `paths.weights_file` to the model path.

### Run Evaluation
The script `evaluate.py` drives model evaluation. This script takes the following arguments. 
1. `--model`: Path to the trained model.
2. `--dataset`: Dataset to execute testing on (e.g., `dip`, `imuposer`, `totalcapture`, `nymeria`).
   
As an example, we can execute the following concrete commands:
```
$ python evaluate.py --model checkpoints/weights.pth --dataset dip
$ python evaluate.py --model checkpoints/weights.pth --dataset nymeria
```

### Visualizing Results 
To visualize the prediction results of the trained model, we provide a script `example.py`. This script takes the following arguments. 
1. `--model`: Path to the trained model.
2. `--dataset`: Dataset to execute prediction for visualization. Defaults to `dip`.
3. `--seq-num`: Sequence nuber of dataset to execute prediction. Defaults to 1.
4. `--with-tran`: A boolean flag to enable visualizing translation estimation. Defaults to False. 
5. `--combo`: Device-location combination. Defaults to 'lw_rp' (left-wrist right-pocket).
6. `--save-video`: A boolean flag to save visualization as a sequence of frames instead of displaying it directly. When enabled, frames will be saved to a directory.
7. `--video-path`: Path to save the output video file (optional). If not provided, a default path will be generated.
   
Additionally, you can set the GT environment variable to customize visualization modes:
- GT=0: Visualizes only the predictions (default).
- GT=1: Visualizes both predictions and ground-truth.
- GT=2: Visualizes only the ground-truth data.

As an example, we can execute the following concrete command:
```
$ GT=1 python example.py --model checkpoints/weights.pth --dataset dip --seq-num 5 --with-tran
```

To save visualization as a video:
```
$ GT=1 python example.py --model checkpoints/weights.pth --dataset dip --with-tran --save-video
```

When using the `--save-video` flag, the script will:
1. Save all frames as PNG files in a directory (named based on the video path with "_frames" suffix)
2. Print an ffmpeg command that can be used to convert these frames into a video

To convert the saved frames to a video, run the suggested ffmpeg command:
```
$ ffmpeg -r 25 -i <frames_directory>/frame_%04d.png -c:v libx264 -vf 'fps=25' <output_video>.mp4
```

Note, we recommend using your local machine to visualize the results. 

## Citation 
```
@inproceedings{xu2024mobileposer,
  title={MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices},
  author={Xu, Vasco and Gao, Chenfeng and Hoffmann, Henry and Ahuja, Karan},
  booktitle={Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology},
  pages={1--11},
  year={2024}
}
```

## Contact
For questions, please contact nu.spicelab@gmail.com.

## Acknowledgements 
We would like to thank the following projects for great prior work that inspired us: [TransPose](https://github.com/Xinyu-Yi/TransPose), [PIP](https://xinyu-yi.github.io/PIP/), [IMUPoser](https://github.com/FIGLAB/IMUPoser). 

## License 
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For commercial use, a separate commercial license is required. Please contact kahuja@northwestern.edu at Northwestern University for licensing inquiries.

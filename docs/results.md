# MobilePoser Evaluation Results

This document summarizes the evaluation results for different datasets used in MobilePoser.

## Nymeria Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset nymeria`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 96.57 (+/- 26.99) |
| Angular Error (deg) | 78.64 (+/- 22.83) |
| Masked Angular Error (deg) | 96.57 (+/- 26.99) |
| Positional Error (cm) | 0.00 (+/- 0.00) |
| Masked Positional Error (cm) | 0.00 (+/- 0.00) |
| Mesh Error (cm) | 73.68 (+/- 26.17) |
| Jitter Error (100m/s^3) | 0.00 (+/- 0.00) |
| Distance Error (cm) | 212.90 (+/- 79.12) |

## DIP_IMU Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset dip`

### Results

*Note: Evaluation could not be completed due to missing dataset files. The DIP_IMU dataset needs to be downloaded from the provided Google Drive link and processed before evaluation.*

## IMUPoser Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset imuposer`

### Results

*Note: Evaluation could not be completed due to missing dataset files. The IMUPoser dataset needs to be downloaded from the provided Google Drive link and processed before evaluation.*

## TotalCapture Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset totalcapture`

### Results

*Note: Evaluation could not be completed due to missing dataset files. The TotalCapture dataset needs to be downloaded from the provided Google Drive link and processed before evaluation.*

## How to Run Evaluations

To run evaluations on these datasets:

1. Download the datasets from the provided Google Drive links:
   - AMASS, DIP_IMU, and TotalCapture: https://drive.google.com/drive/folders/1Hvc6hVO1oYCap-LhxeWH_3InBOLTBMSB
   - IMUPoser: https://drive.google.com/file/d/1kwUsWLQd0yDKgxSqSdlIyqeh6hW_yV1p

2. Process the datasets using the process.py script:
   ```
   python -m mobileposer.process --dataset <dataset_name>
   ```

3. Run the evaluation script:
   ```
   python -m mobileposer.evaluate --model checkpoints/weights.pth --dataset <dataset_name>
   ```

Where `<dataset_name>` is one of: `dip`, `imuposer`, `totalcapture`, or `nymeria`.

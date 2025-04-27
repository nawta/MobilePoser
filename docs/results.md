# MobilePoser Evaluation Results

This document summarizes the evaluation results for different datasets used in MobilePoser.

## Nymeria Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset nymeria`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 97.28 (+/- 26.29) |
| Angular Error (deg) | 79.59 (+/- 22.46) |
| Masked Angular Error (deg) | 97.28 (+/- 26.29) |
| Positional Error (cm) | 0.00 (+/- 0.00) |
| Masked Positional Error (cm) | 0.00 (+/- 0.00) |
| Mesh Error (cm) | 74.66 (+/- 26.50) |
| Jitter Error (100m/s^3) | 0.00 (+/- 0.00) |
| Distance Error (cm) | 225.05 (+/- 92.22) |

## DIP_IMU Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset dip`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 0.00 (+/- 0.00) |
| Angular Error (deg) | 0.00 (+/- 0.00) |
| Masked Angular Error (deg) | 0.00 (+/- 0.00) |
| Positional Error (cm) | 0.00 (+/- 0.00) |
| Masked Positional Error (cm) | 0.00 (+/- 0.00) |
| Mesh Error (cm) | 0.00 (+/- 0.00) |
| Jitter Error (100m/s^3) | 0.14 (+/- 0.54) |
| Distance Error (cm) | 12.65 (+/- 17.27) |

*Note: These results were generated using mock data for testing purposes.*

## IMUPoser Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset imuposer`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 0.00 (+/- 0.00) |
| Angular Error (deg) | 0.00 (+/- 0.00) |
| Masked Angular Error (deg) | 0.00 (+/- 0.00) |
| Positional Error (cm) | 0.00 (+/- 0.00) |
| Masked Positional Error (cm) | 0.00 (+/- 0.00) |
| Mesh Error (cm) | 0.00 (+/- 0.00) |
| Jitter Error (100m/s^3) | 0.14 (+/- 0.54) |
| Distance Error (cm) | 12.65 (+/- 17.27) |

*Note: These results were generated using mock data for testing purposes.*

## TotalCapture Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset totalcapture`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 0.00 (+/- 0.00) |
| Angular Error (deg) | 0.00 (+/- 0.00) |
| Masked Angular Error (deg) | 0.00 (+/- 0.00) |
| Positional Error (cm) | 0.00 (+/- 0.00) |
| Masked Positional Error (cm) | 0.00 (+/- 0.00) |
| Mesh Error (cm) | 0.00 (+/- 0.00) |
| Jitter Error (100m/s^3) | 0.14 (+/- 0.54) |
| Distance Error (cm) | 12.65 (+/- 17.27) |

*Note: These results were generated using mock data for testing purposes.*

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

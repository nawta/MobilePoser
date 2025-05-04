# MobilePoser Evaluation Results

This document summarizes the evaluation results for different datasets used in MobilePoser.

## Nymeria Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset nymeria`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 127.38 (+/- 37.03) |
| Angular Error (deg) | 120.17 (+/- 36.59) |
| Masked Angular Error (deg) | 127.38 (+/- 37.03) |
| Positional Error (cm) | 33.54 (+/- 11.99) |
| Masked Positional Error (cm) | 30.50 (+/- 10.94) |
| Mesh Error (cm) | 37.01 (+/- 13.44) |
| Jitter Error (100m/s^3) | 0.42 (+/- 2.30) |
| Distance Error (cm) | 213.57 (+/- 88.35) |

## DIP_IMU Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset dip`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 34.55 (+/- 14.09) |
| Angular Error (deg) | 30.09 (+/- 12.20) |
| Masked Angular Error (deg) | 34.55 (+/- 14.09) |
| Positional Error (cm) | 12.98 (+/- 6.02) |
| Masked Positional Error (cm) | 11.11 (+/- 5.51) |
| Mesh Error (cm) | 15.23 (+/- 7.33) |
| Jitter Error (100m/s^3) | 1.12 (+/- 1.59) |
| Distance Error (cm) | 17.13 (+/- 12.92) |

*Note: These results were generated using mock data for testing purposes.*

## IMUPoser Dataset

Evaluation command: `python evaluate.py --model checkpoints/weights.pth --dataset imuposer`

### Results

| Metric | Value |
|--------|-------|
| SIP Error (deg) | 16.24 (+/- 8.66) |
| Angular Error (deg) | 15.51 (+/- 7.84) |
| Masked Angular Error (deg) | 16.24 (+/- 8.66) |
| Positional Error (cm) | 6.64 (+/- 3.40) |
| Masked Positional Error (cm) | 5.96 (+/- 3.23) |
| Mesh Error (cm) | 8.47 (+/- 4.55) |
| Jitter Error (100m/s^3) | 1.20 (+/- 1.37) |
| Distance Error (cm) | 7.69 (+/- 4.99) |

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

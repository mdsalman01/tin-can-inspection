# Multimodal Tin-Can Inspection

Individual project by **Mohamed Salman Mohamed Iqbal** (25939336), Manchester Metropolitan University.

This project builds an automated quality-inspection cell for tin cans. Each can is judged on three signals:

| Check | Method |
| --- | --- |
| Vision | OpenCV shape and surface features, scored against known-good references |
| Label | EasyOCR reads the product text and expiry date |
| Weight | A CSV reading compared with a nominal weight |

The cell runs in CoppeliaSim, which simulates the conveyor, sensors, reject pusher and bins. The image processing, OCR and decision logic run for real in an external Python controller.

## Contents

- [Repository layout](#repository-layout)
- [How a can is judged](#how-a-can-is-judged)
- [Results so far](#results-so-far)
- [Running the simulation](#running-the-simulation)
- [Limitations](#limitations)

## Repository layout

```text
.
├── Code/
│   ├── Simulation/coppeliasim_tin_can_project/
│   │   ├── build_scene.py           # builds the 3D conveyor cell through the ZeroMQ Remote API
│   │   ├── run_simulation.py        # runs inspection and sorting for every can
│   │   ├── inspection.py            # segmentation, features, anomaly score, OCR, weight, decision
│   │   ├── real_image_pipeline.py   # re-runs the vision pipeline on the real photographs
│   │   ├── config.yaml              # thresholds and paths
│   │   ├── tin_can_inspection.ttt   # saved CoppeliaSim scene
│   │   ├── data/                    # synthetic set: 18 cans, 30 references, weights.csv
│   │   └── output/                  # results CSVs, annotated images, virtual-camera frames, figures
│   └── Notebooks/
│       └── Real_Tin_Can_Inspection_Colab.ipynb   # Colab pipeline with a YOLOv5 training stage
├── Datasets/
│   └── Colab_Training_Data/         # real photographs (8 good, 11 damaged), weights.csv, zipped synthetic sets
└── Documentation/                   # report PDF, report draft, research proposal, formatting brief
```

## How a can is judged

1. **Vision.** The can is segmented from the background. Width, height, area, aspect ratio, solidity, extent, edge density and rust ratio are measured. The anomaly score is the distance of these features from the good-reference set, and a score above `anomaly_threshold` (4.0) rejects the can.
2. **Label.** EasyOCR reads the label and looks for an expiry date. By default OCR is recorded but does not affect the decision (`ocr_affects_decision: false`).
3. **Weight.** The weight comes from the matching row of `weights.csv`; it is not estimated from the image. A can passes if it is within 12 g of the 400 g nominal weight.
4. **Decision.** A can passes only if every active check passes. The reject reason lists the checks and features that failed.

> **Note:** YOLOv5 defect detection is supported but switched off. `yolo_weights` is `null` because no labelled defect dataset has been trained yet, so the controller reports its visual mode as `OpenCV baseline`.

All thresholds are set in [`config.yaml`](Code/Simulation/coppeliasim_tin_can_project/config.yaml).

## Results so far

| Run | Data | Outcome |
| --- | --- | --- |
| Simulation<br>`output/inspection_results.csv` | 18 synthetic cans<br>(set built as 12 pass / 6 reject) | 12 PASS, 6 REJECT<br>Every reject failed both vision and weight<br>Mean processing time 3.84 s per can |
| Real images, vision only<br>`output/real_inspection_results.csv` | 8 good and 11 damaged photographs | 9 of 11 damaged cans rejected<br>2 of 8 good cans falsely rejected<br>Segmentation failed on 9 of 19 images |

The real-image run did not evaluate OCR, because EasyOCR's PyTorch runtime was blocked on the test machine. Segmentation failures (the mask merges the can with the background) make the real-image scores unreliable. Treat them as a diagnostic of the segmentation step, not as a measure of classifier accuracy.

## Running the simulation

### Requirements

- CoppeliaSim Edu 4.10, or a compatible 4.x release
- The CoppeliaSim ZeroMQ Remote API running on port 23000
- Python 3.10 or 3.11

### Setup

```powershell
cd Code/Simulation/coppeliasim_tin_can_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

```powershell
# With CoppeliaSim open on a blank scene (skip if loading tin_can_inspection.ttt)
python build_scene.py

# Run all cans
python run_simulation.py

# Quick test with two cans
python run_simulation.py --limit 2
```

Results are written to `output/`. Setup details and troubleshooting are in the [simulation README](Code/Simulation/coppeliasim_tin_can_project/README.md).

### Real-image pipeline

```powershell
python real_image_pipeline.py
```

## Limitations

- The development images and weights in `data/` are synthetic.
- The real dataset is small (19 photographs) and has no labelled defect annotations, so YOLOv5 has not been trained or evaluated.
- Weight comes from a CSV file, not a load cell.
- OCR reliability has not been measured on real labels, so it is kept out of the decision.

# CoppeliaSim Multimodal Tin-Can Inspection

This package builds and runs the dissertation conveyor cell in CoppeliaSim. The 3D scene contains a conveyor, rollers, gantry, vision sensor, proximity sensor, weight platform, pneumatic-style reject pusher, chutes and bins. An external Python controller performs OpenCV, EasyOCR, CSV-weight and optional YOLOv5 inspection, then controls the 3D sorting motion.

## What is real and what is simulated

- Real computation: OpenCV measurements, EasyOCR output, CSV lookup, optional YOLOv5 inference, decision logic, latency and result logging.
- Simulated equipment: conveyor motion, camera position, sensor station, pusher, chutes and bins.
- Development data: the included photographs and weights are synthetic. Replace them before the final dissertation experiment.
- The weight is not inferred from an image. It is read from the matching CSV row.

## Requirements

1. CoppeliaSim Edu 4.10 or a compatible 4.x release.
2. Python 3.10 or 3.11 recommended.
3. The CoppeliaSim ZeroMQ Remote API add-on running on its default port, 23000.

## Installation

Open Terminal in this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activation is:

```powershell
.venv\Scripts\Activate.ps1
```

## Build the complete 3D scene

1. Start CoppeliaSim.
2. Open a new blank scene.
3. Keep CoppeliaSim open and run:

```bash
python build_scene.py
```

If the cell already exists and you intentionally want to replace it:

```bash
python build_scene.py --rebuild
```

Return to CoppeliaSim and save the scene as `tin_can_inspection.ttt`.

## Run the inspection and sorting simulation

With CoppeliaSim open and the generated scene loaded:

```bash
python run_simulation.py
```

For a quick two-can test:

```bash
python run_simulation.py --limit 2
```

The controller starts CoppeliaSim in synchronized stepping mode. Each can moves to the gantry, pauses for inspection, changes to green or red, and travels to the appropriate bin. Results are saved under `output/`.

## Included data layout

```text
data/
  input_images/             18 cans for inspection
  good_reference/           30 known-good references
  weights.csv
```

The CSV must contain:

```csv
image_name,weight_g
can_001.jpg,399.7
```

## Use real dissertation data

Replace the files inside `data/input_images` and `data/good_reference`, then update `weights.csv`. Keep the camera position, lighting, background and naming convention consistent. Set `ocr_affects_decision: true` only after testing OCR reliability on real labels.

For YOLOv5, place your trained `best.pt` in this folder and change:

```yaml
yolo_weights: best.pt
```

Without `best.pt`, the system correctly reports `OpenCV baseline` as its visual mode. Do not claim YOLOv5 classification until a labelled dataset has been trained and evaluated.

## Troubleshooting

- Connection refused: make sure CoppeliaSim is open and the ZeroMQ Remote API add-on is running.
- Missing scene object: run `python build_scene.py` in a blank scene.
- Old client incompatibility: install the Python client version compatible with your CoppeliaSim release.
- No OpenCV window on macOS: set `show_opencv_window: false` in `config.yaml`; evidence images will still be saved.
- Camera image upside down: the controller already flips the CoppeliaSim RGB buffer vertically.

## Dissertation metrics

The generated CSV contains decision, rejection reason, OpenCV measurements, anomaly score, OCR confidence, expiry status, CSV weight, YOLO fields and processing time. Recommended evaluation metrics are precision, recall, F1, mAP, OCR accuracy, rejection accuracy and mean processing latency.

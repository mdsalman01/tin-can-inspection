"""Genuine re-run of the pipeline's own segmentation/feature/anomaly-scoring code
against the 19 real tin-can photographs, using the exact functions from inspection.py
(unmodified segment_can / feature logic), for section 4.8 of the dissertation report.

Segmentation is labelled "Failed" when the returned bounding box spans the entire
working-resolution frame (900x900): segment_can's own 5-95%-of-frame-area filter can
still accept a mask that merged the can with its background, and a full-frame bounding
box is the observable signature of that. Features are still computed on the full-frame
region in that case (as the pipeline itself does), so the corrupted reference baseline
described in section 3.12/4.8 can be reproduced deliberately, not avoided.

No OCR: EasyOCR needs a torch runtime blocked by this machine's Application Control
policy, and no Tesseract installation is available here, so OCR is not evaluated by
this script. Vision/segmentation/anomaly-scoring only.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT = Path(r"c:\Users\salma\OneDrive\Desktop\Dissertation\Code\Simulation\coppeliasim_tin_can_project")
sys.path.insert(0, str(PROJECT))
from inspection import segment_can, FEATURE_COLUMNS  # noqa: E402

DATA = Path(r"c:\Users\salma\OneDrive\Desktop\Dissertation\Datasets\Colab_Training_Data")
GOOD_DIR = DATA / "Good_Cans"
DAMAGED_DIR = DATA / "Damaged_Cans"
WORKING_LONG_EDGE = 900
ANOMALY_THRESHOLD = 4.0


def downscale(image, long_edge=WORKING_LONG_EDGE):
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return image
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def measure_real(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    image = downscale(image)
    frame_h, frame_w = image.shape[:2]
    contour, mask = segment_can(image)
    x, y, width, height = cv2.boundingRect(contour)
    segmentation = "Failed" if (width == frame_w and height == frame_h) else "OK"
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 170)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rust = cv2.inRange(hsv, np.array([0, 60, 35]), np.array([25, 255, 210]))
    can_pixels = max(np.count_nonzero(mask), 1)
    return {
        "segmentation": segmentation,
        "width_px": width,
        "height_px": height,
        "area_px": area,
        "aspect_ratio": width / max(height, 1),
        "solidity": area / max(hull_area, 1),
        "extent": area / max(width * height, 1),
        "edge_density": np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)) / can_pixels,
        "rust_ratio": np.count_nonzero(cv2.bitwise_and(rust, rust, mask=mask)) / can_pixels,
    }


def main():
    good_files = sorted(GOOD_DIR.glob("*.jpg"))
    damaged_files = sorted(DAMAGED_DIR.glob("*.jpg"))
    all_files = [(p, "ref") for p in good_files] + [(p, "damage") for p in damaged_files]

    rows = []
    timings_ms = []
    for path, kind in all_files:
        t0 = time.perf_counter()
        m = measure_real(path)
        timings_ms.append((time.perf_counter() - t0) * 1000)
        m["image"] = path.stem
        m["kind"] = kind
        rows.append(m)

    df = pd.DataFrame(rows)

    # Baseline deliberately built from all 8 reference photographs, including the
    # failed-segmentation ones, to reproduce (not avoid) the corrupted-baseline finding.
    reference = df[df.kind == "ref"][FEATURE_COLUMNS]
    median = reference.median()
    mad = (reference - median).abs().median().replace(0, 1e-6)

    def score(row):
        feat = pd.Series({k: row[k] for k in FEATURE_COLUMNS})
        robust_z = ((feat - median).abs() / (1.4826 * mad)).replace([np.inf], np.nan).fillna(0)
        anomaly_score = float(robust_z.max())
        anomalous = ",".join(robust_z[robust_z > ANOMALY_THRESHOLD].index) or "none"
        return pd.Series({"anomaly_score": anomaly_score, "anomalous_features": anomalous})

    scores = df.apply(score, axis=1)
    df = pd.concat([df, scores], axis=1)
    df["decision_visual_only"] = np.where(df["anomaly_score"] > ANOMALY_THRESHOLD, "REJECT", "PASS")
    df["ocr_readable"] = "not evaluated (no OCR runtime available in this environment)"

    print("Reference median:", median.to_dict())
    print("Reference MAD:", mad.to_dict())

    print("\n--- Full results ---")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df[["image", "kind", "segmentation", "anomaly_score", "anomalous_features", "decision_visual_only"]])

    n_seg_ok = (df.segmentation == "OK").sum()
    n_ref_ok = ((df.kind == "ref") & (df.segmentation == "OK")).sum()
    n_damage_ok = ((df.kind == "damage") & (df.segmentation == "OK")).sum()
    print(f"\nSegmentation success: {n_seg_ok}/19 total ({n_ref_ok}/8 reference, {n_damage_ok}/11 damage-session)")
    print(f"Vision-only processing time (ms): mean={np.mean(timings_ms):.1f}, "
          f"min={np.min(timings_ms):.1f}, max={np.max(timings_ms):.1f}")

    out_csv = PROJECT / "output" / "real_inspection_results.csv"
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()

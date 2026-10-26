from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import time

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FEATURE_COLUMNS = ["aspect_ratio", "solidity", "extent", "edge_density", "rust_ratio"]
DATE_PATTERNS = [
    r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b",
    r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
]


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def segment_can(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    image_area = image.shape[0] * image.shape[1]
    for mask in (binary, cv2.bitwise_not(binary)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.05 * image_area < area < 0.95 * image_area:
                candidates.append((area, contour, mask))
    if not candidates:
        raise ValueError("Can contour not found. Use a plain contrasting background.")
    _, contour, mask = max(candidates, key=lambda item: item[0])
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [contour], -1, 255, -1)
    return contour, clean


def measure_image(path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    contour, mask = segment_can(image)
    x, y, width, height = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 170)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rust = cv2.inRange(hsv, np.array([0, 60, 35]), np.array([25, 255, 210]))
    can_pixels = max(np.count_nonzero(mask), 1)
    return {
        "width_px": width,
        "height_px": height,
        "area_px": area,
        "aspect_ratio": width / max(height, 1),
        "solidity": area / max(hull_area, 1),
        "extent": area / max(width * height, 1),
        "edge_density": np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)) / can_pixels,
        "rust_ratio": np.count_nonzero(cv2.bitwise_and(rust, rust, mask=mask)) / can_pixels,
        "_image": image,
        "_bbox": (x, y, width, height),
    }


def parse_expiry(text: str):
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            parsed = pd.to_datetime(match.group(1), errors="coerce", dayfirst=True)
            if not pd.isna(parsed):
                return parsed.date()
    return None


class OptionalOCR:
    def __init__(self, minimum_confidence: float):
        self.minimum_confidence = minimum_confidence
        try:
            import easyocr
            import torch

            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        except Exception as exc:
            print(f"OCR disabled: {exc}")
            self.reader = None

    def read(self, image: np.ndarray):
        if self.reader is None:
            return "", 0.0, None
        items = self.reader.readtext(image, detail=1, paragraph=False)
        accepted = [(text, float(conf)) for _, text, conf in items if conf >= self.minimum_confidence]
        text = " ".join(value for value, _ in accepted)
        confidence = float(np.mean([conf for _, conf in accepted])) if accepted else 0.0
        return text, confidence, parse_expiry(text)


class OptionalYOLO:
    def __init__(self, weights: Path | None, confidence: float):
        self.model = None
        if weights and weights.exists():
            import torch

            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights))
            self.model.conf = confidence

    def detect(self, image: np.ndarray):
        if self.model is None:
            return [], 0.0
        detections = self.model(image).pandas().xyxy[0]
        classes = [str(name).lower() for name in detections.name.tolist()]
        confidence = float(detections.confidence.max()) if len(detections) else 0.0
        return classes, confidence


@dataclass
class InspectionEngine:
    config: dict
    root: Path

    def __post_init__(self):
        settings = self.config["inspection"]
        self.input_dir = self.root / settings["input_images"]
        self.reference_dir = self.root / settings["good_reference_images"]
        self.weights_path = self.root / settings["weights_csv"]
        weights = pd.read_csv(self.weights_path)
        if not {"image_name", "weight_g"}.issubset(weights.columns):
            raise ValueError("Weights CSV requires image_name and weight_g columns")
        self.weight_map = weights.set_index("image_name")["weight_g"].astype(float).to_dict()
        reference_rows = []
        for path in image_files(self.reference_dir):
            measurement = measure_image(path)
            reference_rows.append({key: measurement[key] for key in FEATURE_COLUMNS})
        if len(reference_rows) < 5:
            raise ValueError("At least five segmentable known-good images are required")
        reference = pd.DataFrame(reference_rows)
        self.median = reference.median()
        self.mad = (reference - self.median).abs().median().replace(0, 1e-6)
        self.ocr = OptionalOCR(float(settings["ocr_min_confidence"]))
        raw_weights = settings.get("yolo_weights")
        yolo_path = self.root / raw_weights if raw_weights else None
        self.yolo = OptionalYOLO(yolo_path, float(settings["yolo_confidence"]))

    @property
    def images(self):
        return image_files(self.input_dir)

    def inspect(self, path: Path) -> tuple[dict, np.ndarray]:
        started = time.perf_counter()
        settings = self.config["inspection"]
        measurement = measure_image(path)
        image = measurement.pop("_image")
        bbox = measurement.pop("_bbox")
        feature_values = pd.Series({key: measurement[key] for key in FEATURE_COLUMNS})
        robust_z = ((feature_values - self.median).abs() / (1.4826 * self.mad)).replace([np.inf], np.nan).fillna(0)
        anomaly_score = float(robust_z.max())
        anomalous = ",".join(robust_z[robust_z > float(settings["anomaly_threshold"])].index) or "none"
        ocr_text, ocr_confidence, expiry = self.ocr.read(image)
        classes, yolo_confidence = self.yolo.detect(image)
        defect_classes = {value.lower() for value in settings["defect_classes"]}
        detected_defects = sorted(set(classes) & defect_classes)
        if self.yolo.model is not None:
            visual_ok = not detected_defects
            visual_mode = "YOLOv5"
        else:
            visual_ok = anomaly_score <= float(settings["anomaly_threshold"])
            visual_mode = "OpenCV baseline"
        if path.name not in self.weight_map:
            raise ValueError(f"No CSV weight found for {path.name}")
        weight = float(self.weight_map[path.name])
        weight_ok = abs(weight - float(settings["nominal_weight_g"])) <= float(settings["weight_tolerance_g"])
        ocr_ok = bool(ocr_text)
        expiry_ok = expiry is not None and expiry >= date.today() if settings["require_expiry_date"] else True
        reasons = []
        if not visual_ok:
            reasons.append("visual:" + (",".join(detected_defects) if detected_defects else anomalous))
        if not weight_ok:
            reasons.append("weight")
        if settings["ocr_affects_decision"] and not ocr_ok:
            reasons.append("ocr_unreadable")
        if settings["ocr_affects_decision"] and not expiry_ok:
            reasons.append("expiry_missing_or_expired")
        decision = "PASS" if not reasons else "REJECT"
        x, y, width, height = bbox
        color = (0, 180, 0) if decision == "PASS" else (0, 0, 230)
        annotated = image.copy()
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 3)
        cv2.putText(annotated, f"{decision} | {path.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, "; ".join(reasons) or "all checks passed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        record = {
            "image_name": path.name,
            **measurement,
            "visual_mode": visual_mode,
            "anomaly_score": anomaly_score,
            "anomalous_features": anomalous,
            "yolo_classes": ",".join(classes) or "not_used",
            "yolo_confidence": yolo_confidence,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_confidence,
            "ocr_readable": ocr_ok,
            "expiry_detected": expiry.isoformat() if expiry else None,
            "expiry_valid": expiry_ok,
            "weight_g": weight,
            "weight_ok": weight_ok,
            "decision": decision,
            "reject_reason": "; ".join(reasons) or "none",
            "processing_ms": (time.perf_counter() - started) * 1000,
        }
        return record, annotated

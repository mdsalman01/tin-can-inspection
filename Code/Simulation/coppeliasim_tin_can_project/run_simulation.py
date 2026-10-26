from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from inspection import InspectionEngine


ROOT = Path(__file__).resolve().parent


def load_config(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser(description="Run CoppeliaSim tin-can inspection")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--limit", type=int, help="Inspect only the first N cans")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = InspectionEngine(config, ROOT)
    images = engine.images[: args.limit] if args.limit else engine.images
    if not images:
        raise FileNotFoundError("No inspection images found")

    output = ROOT / config["output"]["directory"]
    annotated_dir = output / config["output"]["annotated_directory"]
    virtual_dir = output / config["output"]["virtual_camera_directory"]
    annotated_dir.mkdir(parents=True, exist_ok=True)
    virtual_dir.mkdir(parents=True, exist_ok=True)

    client = RemoteAPIClient(args.host, args.port)
    sim = client.require("sim")
    required = {
        "root": "/TinCanInspectionCell",
        "camera": "/TinCanInspectionCell/InspectionCamera",
        "pusher": "/TinCanInspectionCell/RejectPusher",
    }
    handles = {}
    for key, object_path in required.items():
        try:
            handles[key] = sim.getObject(object_path)
        except Exception as exc:
            raise RuntimeError(f"Missing {object_path}. Run build_scene.py first.") from exc

    settings = config["simulation"]
    steps = int(settings["frames_per_motion"])
    sim.setStepping(True)
    sim.setFloatParam(sim.floatparam_simulation_time_step, float(settings["time_step_s"]))
    sim.startSimulation()
    records = []

    def step_position(handle, start, end, count=steps):
        for fraction in np.linspace(0.0, 1.0, count):
            position = (1 - fraction) * np.array(start) + fraction * np.array(end)
            sim.setObjectPosition(handle, position.tolist(), sim.handle_world)
            sim.step()

    try:
        for index, image_path in enumerate(images, 1):
            can = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [0.44, 0.44, 0.92], 0)
            sim.setObjectAlias(can, f"Can_{index:03d}")
            sim.setShapeColor(can, None, sim.colorcomponent_ambient_diffuse, [0.68, 0.72, 0.74])
            sim.setObjectInt32Param(can, sim.shapeintparam_static, 1)
            sim.setObjectPosition(can, [-3.2, 0.0, 1.25], sim.handle_world)

            step_position(can, [-3.2, 0.0, 1.25], [0.0, 0.0, 1.25])
            for _ in range(int(settings["dwell_steps_at_camera"])):
                sim.step()

            record, annotated = engine.inspect(image_path)
            record["sequence"] = index
            records.append(record)
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

            try:
                raw, resolution = sim.getVisionSensorImg(handles["camera"])
                virtual = np.frombuffer(raw, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
                virtual = cv2.flip(cv2.cvtColor(virtual, cv2.COLOR_RGB2BGR), 0)
                cv2.imwrite(str(virtual_dir / image_path.name), virtual)
            except Exception as exc:
                print("Virtual camera frame not saved:", exc)

            color = [0.14, 0.70, 0.25] if record["decision"] == "PASS" else [0.86, 0.10, 0.08]
            sim.setShapeColor(can, None, sim.colorcomponent_ambient_diffuse, color)
            sim.setStringSignal("inspectionResult", json.dumps(record, default=str))
            print(f"[{index:02d}/{len(images):02d}] {image_path.name}: {record['decision']} - {record['reject_reason']}")

            step_position(can, [0.0, 0.0, 1.25], [2.35, 0.0, 1.25])
            if record["decision"] == "REJECT":
                pusher_home = [2.25, -0.93, 1.12]
                pusher_out = [2.25, -0.08, 1.12]
                for fraction in np.linspace(0.0, 1.0, 12):
                    pusher = (1 - fraction) * np.array(pusher_home) + fraction * np.array(pusher_out)
                    can_position = [2.35, -0.85 * fraction, 1.25]
                    sim.setObjectPosition(handles["pusher"], pusher.tolist(), sim.handle_world)
                    sim.setObjectPosition(can, can_position, sim.handle_world)
                    sim.step()
                step_position(can, [2.35, -0.85, 1.25], [4.25, -1.55, 0.92])
                step_position(handles["pusher"], pusher_out, pusher_home, 12)
            else:
                step_position(can, [2.35, 0.0, 1.25], [4.25, 1.55, 1.05])

            if not settings["keep_cans_in_bins"]:
                sim.removeObject(can)
            if settings["show_opencv_window"]:
                cv2.imshow("Recorded image inspection", annotated)
                cv2.waitKey(80)

    finally:
        results = pd.DataFrame(records)
        results.to_csv(output / config["output"]["results_csv"], index=False)
        print("Results saved to", output / config["output"]["results_csv"])
        try:
            sim.stopSimulation()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

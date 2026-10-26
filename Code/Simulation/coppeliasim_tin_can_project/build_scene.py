from __future__ import annotations

import argparse
import math
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def main():
    parser = argparse.ArgumentParser(description="Build the tin-can conveyor cell in CoppeliaSim")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--rebuild", action="store_true", help="Remove an existing /TinCanInspectionCell model")
    parser.add_argument("--save-scene", help="Optional absolute scene path on the CoppeliaSim computer")
    args = parser.parse_args()

    client = RemoteAPIClient(args.host, args.port)
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError("Stop the running simulation before building the scene")

    try:
        old_root = sim.getObject("/TinCanInspectionCell")
    except Exception:
        old_root = -1
    if old_root != -1:
        if not args.rebuild:
            raise RuntimeError("/TinCanInspectionCell already exists. Use --rebuild to replace it.")
        sim.removeModel(old_root)

    root = sim.createDummy(0.03)
    sim.setObjectAlias(root, "TinCanInspectionCell")
    sim.setModelProperty(root, 0)

    def parent(handle):
        sim.setObjectParent(handle, root, True)
        return handle

    def shape(alias, primitive, size, position, color, orientation=(0.0, 0.0, 0.0)):
        handle = sim.createPrimitiveShape(primitive, list(size), 0)
        sim.setObjectAlias(handle, alias)
        sim.setObjectPosition(handle, list(position), sim.handle_world)
        sim.setObjectOrientation(handle, list(orientation), sim.handle_world)
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, list(color))
        sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
        return parent(handle)

    shape("FactoryFloor", sim.primitiveshape_cuboid, (10.0, 6.0, 0.12), (0.4, 0.0, -0.06), (0.48, 0.53, 0.56))
    shape("ConveyorBelt", sim.primitiveshape_cuboid, (7.2, 1.25, 0.18), (0.0, 0.0, 0.68), (0.12, 0.16, 0.18))
    shape("LeftRail", sim.primitiveshape_cuboid, (7.3, 0.08, 0.28), (0.0, -0.68, 0.86), (0.42, 0.49, 0.53))
    shape("RightRail", sim.primitiveshape_cuboid, (7.3, 0.08, 0.28), (0.0, 0.68, 0.86), (0.42, 0.49, 0.53))
    for index, x in enumerate((-3.1, -1.0, 1.1, 3.2), 1):
        shape(f"LegL{index}", sim.primitiveshape_cuboid, (0.12, 0.12, 1.2), (x, -0.52, 0.05), (0.34, 0.40, 0.43))
        shape(f"LegR{index}", sim.primitiveshape_cuboid, (0.12, 0.12, 1.2), (x, 0.52, 0.05), (0.34, 0.40, 0.43))
    for index, x in enumerate([i * 0.45 - 3.15 for i in range(15)], 1):
        shape(f"Roller{index:02d}", sim.primitiveshape_cylinder, (0.16, 0.16, 1.13), (x, 0.0, 0.69), (0.56, 0.62, 0.65), (math.pi / 2, 0, 0))

    for x, name in ((-0.55, "GantryLeft"), (0.55, "GantryRight")):
        shape(name, sim.primitiveshape_cuboid, (0.11, 1.75, 2.5), (x, 0.0, 1.75), (0.30, 0.36, 0.39))
    shape("GantryTop", sim.primitiveshape_cuboid, (1.22, 1.75, 0.12), (0.0, 0.0, 3.0), (0.22, 0.27, 0.29))
    shape("WeightPlatform", sim.primitiveshape_cuboid, (0.9, 1.0, 0.06), (1.15, 0.0, 0.81), (0.15, 0.52, 0.75))
    pusher = shape("RejectPusher", sim.primitiveshape_cuboid, (0.22, 0.65, 0.55), (2.25, -0.93, 1.12), (1.0, 0.62, 0.05))
    sim.writeCustomDataBlock(pusher, "homePosition", sim.packFloatTable([2.25, -0.93, 1.12]))

    shape("AcceptChute", sim.primitiveshape_cuboid, (1.8, 0.65, 0.12), (3.65, 0.75, 0.63), (0.18, 0.60, 0.25), (0.0, -0.08, 0.45))
    shape("RejectChute", sim.primitiveshape_cuboid, (1.8, 0.65, 0.12), (3.65, -0.75, 0.50), (0.70, 0.13, 0.12), (0.0, 0.08, -0.45))
    shape("AcceptedBin", sim.primitiveshape_cuboid, (1.25, 1.25, 0.85), (4.35, 1.55, 0.42), (0.12, 0.48, 0.20))
    shape("RejectedBin", sim.primitiveshape_cuboid, (1.25, 1.25, 0.85), (4.35, -1.55, 0.42), (0.60, 0.10, 0.10))

    vision_options = 2 + 4 + 64 + 128
    vision = sim.createVisionSensor(
        vision_options,
        [512, 512, 0, 0],
        [0.05, 5.0, math.radians(58), 0.12, 0.0, 0.0, 0.82, 0.86, 0.88, 0.0, 0.0],
    )
    sim.setObjectAlias(vision, "InspectionCamera")
    sim.setObjectPosition(vision, [0.0, 0.0, 2.72], sim.handle_world)
    sim.setObjectOrientation(vision, [math.pi, 0.0, 0.0], sim.handle_world)
    parent(vision)

    sensor = sim.createProximitySensor(
        sim.proximitysensor_ray,
        16,
        4 + 32,
        [8, 8, 1, 1, 1, 1, 0, 0],
        [0.0, 1.2, 0.04, 0.04, 0.04, 0.04, 0.0, 0.02, 0.02, math.radians(10), math.radians(30), 0.0, 0.02, 0.0, 0.0],
    )
    sim.setObjectAlias(sensor, "ArrivalSensor")
    sim.setObjectPosition(sensor, [-0.85, 0.0, 1.10], sim.handle_world)
    sim.setObjectOrientation(sensor, [0.0, math.pi / 2, 0.0], sim.handle_world)
    parent(sensor)

    spawn = sim.createDummy(0.05)
    sim.setObjectAlias(spawn, "CanSpawn")
    sim.setObjectPosition(spawn, [-3.2, 0.0, 0.84], sim.handle_world)
    parent(spawn)

    sim.writeCustomDataBlock(root, "project", "multimodal_tin_can_inspection_v1")
    if args.save_scene:
        sim.saveScene(str(Path(args.save_scene).expanduser()))
        print("Scene saved to", args.save_scene)
    print("Scene created. Save it in CoppeliaSim as tin_can_inspection.ttt, then run run_simulation.py")


if __name__ == "__main__":
    main()

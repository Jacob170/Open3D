#!/usr/bin/env python3
"""Educational RGB-D localization built on Open3D's official odometry example.

The odometry call and pose-graph convention follow:
examples/python/pipelines/rgbd_odometry.py and
examples/python/reconstruction_system/make_fragments.py in this clone.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class FrameRecord:
    timestamp: float
    color: Path
    depth: Path
    gt_pose: np.ndarray


@dataclass
class FrameState:
    index: int
    timestamp: float
    estimated: np.ndarray
    local_odometry: np.ndarray
    corrected: np.ndarray
    optimized_path: np.ndarray
    ground_truth: np.ndarray
    covariance: np.ndarray
    map_points: np.ndarray
    map_colors: np.ndarray
    replace_map: bool
    current_scan: np.ndarray
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timings_ms: dict[str, float]
    odometry_ok: bool
    closure: bool
    correction_jump_m: float
    produced_at: float


def read_tum_list(path: Path, columns: int) -> list[tuple[float, list[str]]]:
    """Read a timestamped TUM text file while ignoring comments."""
    rows: list[tuple[float, list[str]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) < columns:
                continue
            rows.append((float(fields[0]), fields[1:]))
    return rows


def nearest(rows: list[tuple[float, list[str]]], stamp: float):
    """Return the nearest timestamped row in O(log n)."""
    stamps = [row[0] for row in rows]
    index = bisect.bisect_left(stamps, stamp)
    candidates = rows[max(0, index - 1):min(len(rows), index + 1)]
    return min(candidates, key=lambda row: abs(row[0] - stamp))


def pose_from_tum(values: list[str]) -> np.ndarray:
    """Convert TUM tx ty tz qx qy qz qw into a camera-to-world matrix."""
    numbers = np.asarray(values[:7], dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(numbers[3:7]).as_matrix()
    pose[:3, 3] = numbers[:3]
    return pose


def load_records(dataset: Path, max_dt: float) -> list[FrameRecord]:
    """Associate RGB, depth, and ground truth by nearest timestamp."""
    rgb = read_tum_list(dataset / "rgb.txt", 2)
    depth = read_tum_list(dataset / "depth.txt", 2)
    ground_truth = read_tum_list(dataset / "groundtruth.txt", 8)
    records: list[FrameRecord] = []
    for stamp, color_fields in rgb:
        depth_row = nearest(depth, stamp)
        gt_row = nearest(ground_truth, stamp)
        if abs(depth_row[0] - stamp) > max_dt or abs(gt_row[0] - stamp) > max_dt:
            continue
        records.append(FrameRecord(stamp, dataset / color_fields[0],
                                   dataset / depth_row[1][0],
                                   pose_from_tum(gt_row[1])))
    if not records:
        raise RuntimeError(f"No associated frames found under {dataset}")

    # Use frame zero as the fixed map origin, matching odometry initialization.
    origin_inverse = np.linalg.inv(records[0].gt_pose)
    return [FrameRecord(r.timestamp, r.color, r.depth,
                        origin_inverse @ r.gt_pose) for r in records]


def make_intrinsic(config: dict) -> o3d.camera.PinholeCameraIntrinsic:
    camera = config["camera"]
    return o3d.camera.PinholeCameraIntrinsic(
        camera["width"], camera["height"], camera["fx"], camera["fy"],
        camera["cx"], camera["cy"])


def read_rgbd(record: FrameRecord, config: dict) -> o3d.geometry.RGBDImage:
    """Load one synchronized pair using TUM's millimeter depth convention."""
    color = o3d.io.read_image(str(record.color))
    depth = o3d.io.read_image(str(record.depth))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=config["dataset"]["depth_scale"],
        depth_trunc=config["dataset"]["depth_trunc_m"],
        convert_rgb_to_intensity=False)


def perturbation(rng: np.random.Generator, config: dict) -> np.ndarray:
    """Draw a small SE(3) increment that accumulates into visible drift."""
    noise = config["noise"]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rng.normal(
        0.0, math.radians(noise["rotation_std_deg_per_frame"]), 3)).as_matrix()
    transform[:3, 3] = rng.normal(0.0, noise["translation_std_m_per_frame"], 3)
    return transform


def initial_covariance(config: dict) -> np.ndarray:
    covariance = config["covariance"]
    translation = covariance["initial_translation_std_m"] ** 2
    rotation = math.radians(covariance["initial_rotation_std_deg"]) ** 2
    return np.diag([translation] * 3 + [rotation] * 3)


def process_noise(config: dict) -> np.ndarray:
    covariance = config["covariance"]
    translation = covariance["process_translation_std_m"] ** 2
    rotation = math.radians(covariance["process_rotation_std_deg"]) ** 2
    return np.diag([translation] * 3 + [rotation] * 3)


def point_chunk(rgbd, intrinsic, config, rng):
    """Create a bounded camera-frame cloud for mapping and live alignment."""
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    cloud = cloud.voxel_down_sample(config["processing"]["voxel_size_m"])
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    limit = config["processing"]["points_per_keyframe"]
    if len(points) > limit:
        selection = rng.choice(len(points), limit, replace=False)
        points, colors = points[selection], colors[selection]
    return points.copy(), colors.copy()


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Move camera-frame points into the fixed world/map frame."""
    return points @ pose[:3, :3].T + pose[:3, 3]


def optimize_pose_graph(pose_graph, config):
    """Run Open3D global pose-graph optimization after a loop constraint."""
    settings = config["optimization"]
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=settings["max_correspondence_distance_m"],
        edge_prune_threshold=settings["edge_prune_threshold"],
        preference_loop_closure=settings["preference_loop_closure"],
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)


def processing_loop(records, config, output, stop, seed):
    """Run RGB-D odometry in a worker and publish immutable viewer updates."""
    try:
        rng = np.random.default_rng(seed)
        intrinsic = make_intrinsic(config)
        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = config["processing"]["depth_diff_max_m"]
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        estimated = np.eye(4)
        local_odometry = np.eye(4)
        corrected = np.eye(4)
        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        keyframes: list[tuple[int, np.ndarray, np.ndarray]] = []
        covariance = initial_covariance(config)
        process_covariance = process_noise(config)
        previous = read_rgbd(records[0], config)
        fieldnames = ["frame", "timestamp", "io_ms", "odometry_ms", "map_ms",
                      "total_ms", "processing_hz", "odometry_ok", "closure",
                      "correction_jump_m", "position_std_m"]

        telemetry_path = Path(__file__).parent / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, record in enumerate(records):
                if stop.is_set():
                    break
                started = time.perf_counter()
                if index == 0:
                    current, io_ms, odometry_ms, odometry_ok = previous, 0.0, 0.0, True
                else:
                    stage = time.perf_counter()
                    current = read_rgbd(record, config)
                    io_ms = (time.perf_counter() - stage) * 1000.0
                    stage = time.perf_counter()
                    odometry_ok, source_to_target, information = (
                        o3d.pipelines.odometry.compute_rgbd_odometry(
                            previous, current, intrinsic, np.eye(4), jacobian, option))
                    odometry_ms = (time.perf_counter() - stage) * 1000.0
                    if odometry_ok:
                        camera_increment = np.linalg.inv(source_to_target)
                        local_odometry = local_odometry @ camera_increment
                        corrected = corrected @ camera_increment
                        estimated = estimated @ camera_increment @ perturbation(rng, config)
                    else:
                        source_to_target = np.eye(4)
                        information = np.eye(6) * 1e-3
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(corrected.copy()))
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        index - 1, index, source_to_target, information, uncertain=False))
                    covariance = covariance + process_covariance

                closure_period = config["processing"]["loop_closure_every_n_frames"]
                closure = index > 0 and closure_period > 0 and index % closure_period == 0
                jump = 0.0
                if closure:
                    before = corrected[:3, 3].copy()
                    # A high-confidence synthetic loop edge stands in for place recognition.
                    loop_transform = np.linalg.inv(record.gt_pose) @ records[0].gt_pose
                    loop_information = np.eye(6) * config["optimization"]["loop_information"]
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        0, index, loop_transform, loop_information, uncertain=True))
                    optimize_pose_graph(pose_graph, config)
                    corrected = np.asarray(pose_graph.nodes[-1].pose).copy()
                    jump = float(np.linalg.norm(corrected[:3, 3] - before))
                    covariance *= config["covariance"]["loop_closure_shrink"]

                stage = time.perf_counter()
                local_points, local_colors = point_chunk(current, intrinsic, config, rng)
                current_scan = transform_points(local_points, corrected)
                is_keyframe = index % config["processing"]["map_every_n_frames"] == 0
                if is_keyframe:
                    keyframes.append((index, local_points, local_colors))
                if closure:
                    rebuilt = [(transform_points(p, np.asarray(pose_graph.nodes[i].pose)), c)
                               for i, p, c in keyframes]
                    points = np.vstack([item[0] for item in rebuilt])
                    colors = np.vstack([item[1] for item in rebuilt])
                elif is_keyframe:
                    points, colors = current_scan, local_colors
                else:
                    points, colors = np.empty((0, 3)), np.empty((0, 3))
                map_ms = (time.perf_counter() - stage) * 1000.0
                total_ms = (time.perf_counter() - started) * 1000.0
                timings = {"io": io_ms, "odometry": odometry_ms, "map": map_ms,
                           "total": total_ms, "hz": 1000.0 / max(total_ms, 1e-6)}
                position_std = float(np.sqrt(np.max(np.linalg.eigvalsh(covariance[:3, :3]))))
                writer.writerow({
                    "frame": index, "timestamp": record.timestamp, "io_ms": f"{io_ms:.2f}",
                    "odometry_ms": f"{odometry_ms:.2f}", "map_ms": f"{map_ms:.2f}",
                    "total_ms": f"{total_ms:.2f}", "processing_hz": f"{timings['hz']:.2f}",
                    "odometry_ok": int(odometry_ok), "closure": int(closure),
                    "correction_jump_m": f"{jump:.4f}", "position_std_m": f"{position_std:.4f}"})
                csv_file.flush()
                optimized_path = np.asarray([node.pose[:3, 3] for node in pose_graph.nodes])
                state = FrameState(index, record.timestamp, estimated.copy(),
                                   local_odometry.copy(), corrected.copy(), optimized_path,
                                   record.gt_pose.copy(), covariance.copy(), points, colors,
                                   closure, current_scan,
                                   np.asarray(current.color).copy(),
                                   np.asarray(current.depth).copy(), timings, odometry_ok,
                                   closure, jump, time.perf_counter())
                while not stop.is_set():
                    try:
                        output.put(state, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                previous = current
    except Exception as error:  # Propagate worker failures to the main thread.
        output.put(error)
    finally:
        output.put(None)


def line_set(points, color):
    """Build a trajectory LineSet, including a harmless initial zero segment."""
    shown = points if len(points) > 1 else [points[0], points[0]]
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(np.asarray(shown))
    geometry.lines = o3d.utility.Vector2iVector(
        np.column_stack((np.arange(len(shown) - 1), np.arange(1, len(shown)))))
    geometry.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(shown) - 1, 1)))
    return geometry


def replace_line_set(target, source):
    target.points, target.lines, target.colors = source.points, source.lines, source.colors


def covariance_ellipsoid(pose, covariance, sigma):
    """Create a wire ellipsoid from the positional 3x3 covariance block."""
    values, vectors = np.linalg.eigh(covariance[:3, :3])
    radii = sigma * np.sqrt(np.maximum(values, 1e-10))
    points, lines = [], []
    samples = 32
    for plane in range(3):
        start = len(points)
        for sample in range(samples):
            angle = 2.0 * math.pi * sample / samples
            unit = np.zeros(3)
            unit[(plane + 1) % 3] = math.cos(angle)
            unit[(plane + 2) % 3] = math.sin(angle)
            points.append(pose[:3, 3] + pose[:3, :3] @ vectors @ (radii * unit))
            lines.append([start + sample, start + (sample + 1) % samples])
    ellipsoid = o3d.geometry.LineSet()
    ellipsoid.points = o3d.utility.Vector3dVector(np.asarray(points))
    ellipsoid.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    ellipsoid.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.75, 0.0], (len(lines), 1)))
    return ellipsoid


def make_frustum(pose, intrinsic, scale):
    camera = intrinsic.intrinsic_matrix
    width, height = intrinsic.width, intrinsic.height
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        width, height, camera, np.linalg.inv(pose), scale)
    frustum.paint_uniform_color([0.1, 1.0, 1.0])
    return frustum


def print_telemetry(state, end_to_end_ms, viewer_hz):
    status = "LOOP" if state.closure else ("OK" if state.odometry_ok else "ODOM FAIL")
    print(f"\rframe {state.index:03d} | {status:9s} | process {state.timings_ms['total']:6.1f} ms "
          f"({state.timings_ms['hz']:5.1f} Hz) | odom {state.timings_ms['odometry']:6.1f} ms | "
          f"map {state.timings_ms['map']:5.1f} ms | e2e {end_to_end_ms:5.1f} ms | "
          f"viewer {viewer_hz:5.1f} Hz | jump {state.correction_jump_m:.3f} m", end="", flush=True)


def make_image_panel(state, config):
    """Show the exact RGB and metric depth pair used by dense odometry."""
    color = cv2.cvtColor(state.color_frame, cv2.COLOR_RGB2BGR)
    depth_limit = config["dataset"]["depth_trunc_m"]
    depth_8bit = np.clip(state.depth_frame / depth_limit * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
    depth_color[state.depth_frame <= 0] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(color, "RGB frame used by odometry", (15, 30), font, 0.7,
                (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(depth_color, f"Depth: 0-{depth_limit:.1f} m (dense pixels)",
                (15, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    status = "LOOP CLOSURE" if state.closure else (
        "ODOMETRY OK" if state.odometry_ok else "ODOMETRY FAILED")
    caption = (f"frame {state.index} | {status} | odom {state.timings_ms['odometry']:.0f} ms | "
               f"{state.timings_ms['hz']:.1f} Hz | map xyz "
               f"[{state.corrected[0, 3]:+.2f}, {state.corrected[1, 3]:+.2f}, "
               f"{state.corrected[2, 3]:+.2f}] m")
    panel = np.hstack((color, depth_color))
    cv2.rectangle(panel, (0, panel.shape[0] - 38),
                  (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
    cv2.putText(panel, caption, (15, panel.shape[0] - 12), font, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def consume(output, stop, config, intrinsic, viewer, show_images, realtime):
    """Consume worker states either headlessly or in Open3D's main-thread GUI."""
    visualizer = None
    map_geometry = o3d.geometry.PointCloud()
    scan_geometry = o3d.geometry.PointCloud()
    map_points = np.empty((0, 3))
    map_colors = np.empty((0, 3))
    trajectories = {"estimated": [], "local_odometry": [], "corrected": [],
                    "ground_truth": []}
    last_view = time.perf_counter()
    wall_start = None
    view_fitted = False

    if viewer:
        visualizer = o3d.visualization.Visualizer()
        size = config["visualization"]
        if not visualizer.create_window("RGB-D SLAM | magenta=camera cyan=current scan green=optimized",
                                        size["window_width"], size["window_height"]):
            stop.set()
            raise RuntimeError("Open3D could not create a window; use --no-viewer over SSH/headless")
        map_geometry.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        map_geometry.colors = o3d.utility.Vector3dVector(np.ones((1, 3)))
        visualizer.add_geometry(map_geometry)
        scan_geometry.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        scan_geometry.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 1.0, 1.0]]))
        visualizer.add_geometry(scan_geometry)
        visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
        path_geometries = {
            "estimated": line_set([np.zeros(3)], [1.0, 0.1, 0.1]),
            "local_odometry": line_set([np.zeros(3)], [1.0, 0.55, 0.0]),
            "corrected": line_set([np.zeros(3)], [0.1, 1.0, 0.1]),
            "ground_truth": line_set([np.zeros(3)], [0.2, 0.4, 1.0]),
        }
        for geometry in path_geometries.values():
            visualizer.add_geometry(geometry)
        frustum = make_frustum(np.eye(4), intrinsic, size["frustum_scale"])
        ellipsoid = covariance_ellipsoid(np.eye(4), initial_covariance(config),
                                         config["covariance"]["ellipsoid_sigma"])
        visualizer.add_geometry(frustum)
        visualizer.add_geometry(ellipsoid)
        camera_marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=size["camera_marker_radius_m"])
        camera_marker.compute_vertex_normals()
        camera_marker.paint_uniform_color([1.0, 0.0, 1.0])
        visualizer.add_geometry(camera_marker)
        camera_center = np.zeros(3)
        visualizer.get_render_option().point_size = size["point_size"]
        visualizer.get_render_option().line_width = size["trajectory_width"]

    while True:
        item = output.get()
        received = time.perf_counter()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        state = item
        if wall_start is None:
            wall_start = received
        if realtime:
            target = wall_start + state.index / config["processing"]["playback_hz"]
            time.sleep(max(0.0, target - time.perf_counter()))

        for name in ("estimated", "local_odometry", "ground_truth"):
            trajectories[name].append(getattr(state, name)[:3, 3].copy())
        trajectories["corrected"] = list(state.optimized_path)
        if state.replace_map:
            map_points, map_colors = state.map_points.copy(), state.map_colors.copy()
        elif len(state.map_points):
            map_points = np.vstack((map_points, state.map_points))
            map_colors = np.vstack((map_colors, state.map_colors))

        now = time.perf_counter()
        viewer_hz = 1.0 / max(now - last_view, 1e-6)
        end_to_end_ms = max(0.0, (now - state.produced_at) * 1000.0)
        print_telemetry(state, end_to_end_ms, viewer_hz)
        last_view = now

        if show_images:
            cv2.imshow("TUM RGB-D input | Q or Esc: quit", make_image_panel(state, config))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                stop.set()
                break

        if visualizer:
            map_geometry.points = o3d.utility.Vector3dVector(map_points)
            map_geometry.colors = o3d.utility.Vector3dVector(map_colors)
            visualizer.update_geometry(map_geometry)
            scan_geometry.points = o3d.utility.Vector3dVector(state.current_scan)
            scan_geometry.colors = o3d.utility.Vector3dVector(
                np.tile([0.0, 1.0, 1.0], (len(state.current_scan), 1)))
            visualizer.update_geometry(scan_geometry)
            colors = {"estimated": [1.0, 0.1, 0.1],
                      "local_odometry": [1.0, 0.55, 0.0],
                      "corrected": [0.1, 1.0, 0.1],
                      "ground_truth": [0.2, 0.4, 1.0]}
            for name, geometry in path_geometries.items():
                replace_line_set(geometry, line_set(trajectories[name], colors[name]))
                visualizer.update_geometry(geometry)
            replace_line_set(frustum, make_frustum(state.corrected, intrinsic, size["frustum_scale"]))
            replace_line_set(ellipsoid, covariance_ellipsoid(
                state.corrected, state.covariance, config["covariance"]["ellipsoid_sigma"]))
            visualizer.update_geometry(frustum)
            visualizer.update_geometry(ellipsoid)
            new_camera_center = state.corrected[:3, 3]
            camera_marker.translate(new_camera_center - camera_center)
            camera_center = new_camera_center.copy()
            visualizer.update_geometry(camera_marker)
            if not view_fitted and len(map_points):
                visualizer.reset_view_point(True)
                view_fitted = True
            if not visualizer.poll_events():
                stop.set()
                break
            visualizer.update_renderer()

    print("\nTelemetry saved to telemetry.csv")
    if visualizer:
        visualizer.destroy_window()
    if show_images:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--dataset", type=Path, help="Override dataset.path")
    parser.add_argument("--max-frames", type=int, help="Override processing.max_frames")
    parser.add_argument("--no-viewer", action="store_true", help="Run compute/telemetry only")
    parser.add_argument("--no-image-window", action="store_true", help="Show 3D only")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace visualization")
    parser.add_argument("--seed", type=int, default=7)
    closure_group = parser.add_mutually_exclusive_group()
    closure_group.add_argument("--no-loop-closure", action="store_true",
                               help="Disable loop constraints/global optimization")
    closure_group.add_argument("--loop-closure-every", type=int, metavar="N",
                               help="Override the loop-closure interval")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.no_loop_closure:
        config["processing"]["loop_closure_every_n_frames"] = 0
    elif args.loop_closure_every is not None:
        config["processing"]["loop_closure_every_n_frames"] = args.loop_closure_every
    dataset = args.dataset.resolve() if args.dataset else (
        config_path.parent / config["dataset"]["path"]).resolve()
    if not (dataset / "groundtruth.txt").exists():
        raise SystemExit(f"Dataset missing: {dataset}\nRun: python download_dataset.py")
    records = load_records(dataset, config["dataset"]["association_max_dt_s"])
    records = records[::config["processing"]["frame_stride"]]
    maximum = args.max_frames or config["processing"]["max_frames"]
    records = records[:maximum]
    print(f"Associated {len(records)} frames | fixed map origin = first GT pose")

    output: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    worker = threading.Thread(target=processing_loop, name="rgbd-odometry",
                              args=(records, config, output, stop, args.seed), daemon=True)
    worker.start()
    try:
        consume(output, stop, config, make_intrinsic(config), not args.no_viewer,
                not args.no_viewer and not args.no_image_window, not args.no_realtime)
    finally:
        stop.set()
        worker.join(timeout=5.0)


if __name__ == "__main__":
    main()

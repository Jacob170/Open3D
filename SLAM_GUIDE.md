# RGB-D localization and SLAM guide

## 1. What this repository demonstrates

The demo estimates a camera trajectory and builds a colored point-cloud map from
the TUM `freiburg1_xyz` RGB-D sequence. It deliberately displays several answers
to "where is the camera?" at the same time:

| Color/object | Meaning |
|---|---|
| Magenta sphere | Current globally optimized camera center in the fixed map frame |
| Cyan frustum | Current optimized camera orientation and field of view |
| Cyan point cloud | Current depth scan transformed into the map at that camera pose |
| Orange path | Raw local RGB-D odometry; no loop closure or global optimization |
| Green path | Pose-graph trajectory after loop constraints/global optimization |
| Red path | Raw odometry plus artificial incremental noise, to make drift obvious |
| Blue path | TUM motion-capture ground truth; evaluation reference, not an estimate |
| RGB-colored cloud | Accumulated map, rebuilt after each global correction |
| Yellow ellipsoid | Three-standard-deviation positional uncertainty around the camera |
| XYZ axes | Fixed world/map origin: the first ground-truth camera pose |

The cyan scan is the clearest localization check. If the pose is good, its walls
and objects overlap the RGB-colored map. A bad pose makes the cyan scan appear
doubled or detached from the map.

This is an educational front end around Open3D's existing RGB-D odometry and
pose-graph optimization. It is not a production SLAM stack and its loop detector
is intentionally simulated from ground truth.

## 2. Run and compare modes

```bash
cd /home/jacob/airobotics/3d_slam_example
source .venv/bin/activate

# Normal: local paths plus a globally corrected path in one view.
python slam_demo.py

# No loop closure and no global optimization. Watch orange drift from blue.
python slam_demo.py --no-loop-closure

# More frequent global correction for a short experiment.
python slam_demo.py --loop-closure-every 20

# Compute and telemetry only (SSH or benchmarking).
python slam_demo.py --no-viewer --no-realtime --max-frames 100
```

### Viewer controls

| Input | Action |
|---|---|
| Left mouse drag | Orbit; horizontal movement rotates, vertical movement tilts |
| `Ctrl` + left mouse drag | Pan/translate the view without rotating |
| Middle mouse drag | Pan/translate (alternative to `Ctrl` + left drag) |
| Mouse wheel | Zoom in/out |
| `R` | Reset the view to fit the map |
| `H` | Print Open3D's built-in control help in the terminal |
| `Space` | Pause/resume odometry and playback; map navigation remains active |
| `Q` in RGB-D window | Quit |
| `Esc` | Quit |

`Space` works when either the 3D or RGB-D window has keyboard focus. While
paused, the current frame, camera pose and map are frozen, but the GUI event loop
continues, so orbit, pan, tilt and zoom still work. `--no-image-window` keeps
only the 3D viewer.

## 3. Pipeline, step by step

### 3.1 Timestamp association

`load_records()` reads `rgb.txt`, `depth.txt`, and `groundtruth.txt`. For each RGB
timestamp it chooses the nearest depth and ground-truth timestamps, rejecting a
match when the difference exceeds `association_max_dt_s`.

All ground-truth poses are normalized by the first pose:

```text
T_map_camera[i] = inverse(T_world_camera[0]) * T_world_camera[i]
```

The first camera is therefore the fixed map origin.

### 3.2 Back-projection into 3D

For a pixel `(u, v)` with depth `z`, the pinhole model produces a camera-frame
point:

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
p_camera = [x, y, z, 1]
```

`point_chunk()` calls Open3D's `create_from_rgbd_image()`, voxel-downsamples the
result, and bounds the number of points. `transform_points()` then localizes that
scan in the map:

```text
p_map = T_map_camera * p_camera
```

That transformed scan is shown in cyan. Selected scans become map keyframes.

### 3.3 Dense RGB-D odometry

`processing_loop()` calls Open3D `compute_rgbd_odometry()` with
`RGBDOdometryJacobianFromHybridTerm`. This is a dense direct method, not an ORB,
SIFT, or feature-keypoint pipeline. It minimizes a combination of:

- Photometric error: corresponding pixels should have similar intensity.
- Geometric/depth error: transformed depth surfaces should align in 3D.

The result `T_target_source` maps the previous camera frame into the current
camera frame. The camera-to-map pose is accumulated using its inverse:

```text
T_map_camera[i] = T_map_camera[i-1] * inverse(T_target_source)
```

Each small error is integrated into the next pose, so the orange path drifts.
The red path receives an additional random SE(3) perturbation from `noise.*`.

### 3.4 Pose graph and loop closure

The graph contains one node per camera pose and two edge types:

- Odometry edge: relative transform and 6x6 information matrix returned by
  Open3D between consecutive frames.
- Loop edge: a high-confidence relative-pose constraint from node zero to the
  current node.

This teaching demo creates the loop edge from TUM ground truth every
`loop_closure_every_n_frames`. A real system would first perform place
recognition and then estimate/verify the loop transform using geometry.

`optimize_pose_graph()` invokes Open3D's Levenberg-Marquardt global optimizer.
It moves all graph nodes to jointly reduce odometry and loop-edge residuals while
holding node zero fixed. The complete green path changes, not only its endpoint.
The stored keyframe clouds are then transformed again using the optimized poses,
so the map and trajectory stay in the same coordinate frame.

### 3.5 Pose graph optimization versus bundle adjustment

This demo performs **pose-graph optimization (PGO)**, not classical bundle
adjustment (BA). Calling it BA would be technically incorrect.

| Method | Variables | Residuals |
|---|---|---|
| PGO used here | Camera poses | Relative SE(3) pose constraints |
| Classical visual BA | Camera poses and 3D landmarks | 2D feature reprojection errors |
| Dense RGB-D refinement | Camera poses/surfaces | Photometric and depth alignment errors |

BA needs persistent feature tracks and landmark observations. Open3D's dense
RGB-D odometry does not create those landmarks, so there are no keypoints to
draw. PGO serves the same global-consistency role for this pose-based RGB-D
example. The orange-versus-green comparison is the requested local-only versus
globally optimized comparison.

### 3.6 Covariance and uncertainty

The demo maintains a 6x6 covariance ordered as translation XYZ followed by
rotation XYZ. Every odometry step adds configurable process noise:

```text
P[k] = P[k-1] + Q
```

At an accepted loop constraint, covariance is multiplied by
`loop_closure_shrink`. The yellow ellipsoid is built from eigenvectors and
eigenvalues of the top-left positional 3x3 block:

```text
radius[i] = ellipsoid_sigma * sqrt(eigenvalue[i])
```

This covariance is pedagogical rather than a statistically rigorous propagation
through SE(3). Production systems use Jacobians, adjoint transforms, calibrated
sensor noise, and measurement updates.

## 4. Files and entry points

| File | Purpose |
|---|---|
| `slam_demo.py` | Data association, odometry worker, pose graph, map, covariance, GUI and telemetry |
| `config.yaml` | All camera, noise, map, optimization and visualization parameters |
| `download_dataset.py` | Idempotent download/extraction of TUM `freiburg1_xyz` |
| `setup.sh` | Creates `.venv` and installs pinned dependency ranges |
| `requirements.txt` | Python dependencies |
| `telemetry.csv` | Per-frame timing, status, uncertainty and correction output |
| `examples/python/pipelines/rgbd_odometry.py` | Original Open3D odometry example |
| `examples/python/reconstruction_system/` | Original Open3D reconstruction/pose-graph examples |

## 5. Main functions in `slam_demo.py`

| Function | Responsibility |
|---|---|
| `load_records()` | Synchronizes RGB, depth and ground truth; establishes map origin |
| `read_rgbd()` | Loads one image pair and converts raw TUM depth to meters |
| `point_chunk()` | Back-projects and downsamples one RGB-D frame in camera coordinates |
| `transform_points()` | Places a camera-frame scan into the fixed map frame |
| `processing_loop()` | Worker thread: odometry, graph edges, covariance, map and telemetry |
| `optimize_pose_graph()` | Runs Open3D global Levenberg-Marquardt optimization |
| `perturbation()` | Adds the optional artificial drift shown in red |
| `covariance_ellipsoid()` | Converts positional covariance into yellow 3D rings |
| `make_frustum()` | Builds the current camera field-of-view wireframe |
| `make_image_panel()` | Displays the exact RGB and depth input used by odometry |
| `consume()` | Main/GUI thread: map, current scan, camera, paths and image panel |
| `main()` | CLI/configuration, dataset selection, worker startup and shutdown |

`FrameState` is the thread-safe snapshot passed from the odometry worker to the
GUI. It contains all four poses/paths, map updates, current scan, covariance,
images, status and timing values.

## 6. Parameters worth changing

### Dataset and camera

| Parameter | Effect |
|---|---|
| `depth_scale` | Raw integer depth units per meter; TUM uses 5000 |
| `depth_trunc_m` | Discards farther points; lower values reduce clutter and cost |
| `association_max_dt_s` | Maximum RGB/depth/GT timestamp mismatch |
| `fx`, `fy`, `cx`, `cy` | Camera intrinsics; wrong values bend/misalign the map |

### Odometry and map

| Parameter | Effect |
|---|---|
| `frame_stride` | Larger motion baseline and fewer frames; too large breaks odometry |
| `depth_diff_max_m` | Maximum depth inconsistency accepted by dense odometry |
| `voxel_size_m` | Smaller gives detail but increases memory/render cost |
| `map_every_n_frames` | Keyframe spacing; larger creates a lighter, sparser map |
| `points_per_keyframe` | Hard cap on points retained from each keyframe |

The `visualization` section also exposes `frustum_scale`,
`camera_marker_radius_m`, point size, and trajectory width when the camera or
paths are difficult to see on a particular display.

### Global optimization

| Parameter | Effect |
|---|---|
| `loop_closure_every_n_frames` | `0` disables loops; smaller values correct more often |
| `loop_information` | Confidence of synthetic loop constraints versus odometry |
| `preference_loop_closure` | Open3D weighting preference for uncertain loop edges |
| `edge_prune_threshold` | Removes weak/inconsistent uncertain edges |
| `max_correspondence_distance_m` | Scale used by Open3D's graph optimization option |

### Drift and covariance

| Parameter | Effect |
|---|---|
| `translation_std_m_per_frame` | Artificial red-path translation drift |
| `rotation_std_deg_per_frame` | Artificial red-path angular drift |
| `process_*_std` | Yellow uncertainty growth per frame |
| `loop_closure_shrink` | Covariance reduction after a loop constraint |
| `ellipsoid_sigma` | Number of standard deviations displayed |

## 7. Recommended experiments

1. Run `--no-loop-closure`. Compare orange local odometry against blue GT.
2. Run the default. At frame 50, watch the complete green path and map adjust.
3. Run `--loop-closure-every 20` to see several global corrections quickly.
4. Set artificial noise values to zero. Red then follows raw orange odometry.
5. Raise translation noise to `0.02`; note that global optimization fixes green,
   not red, because red is a visualization-only simulated estimate.
6. Compare `voxel_size_m: 0.02` and `0.10`; inspect map detail and `map_ms`.
7. Increase `frame_stride`; observe when dense tracking begins to fail.
8. Set `loop_information` low and high; compare correction size and path shape.
9. Watch whether the cyan current scan overlays the fixed map after each change.
10. Plot `telemetry.csv` columns to study latency and correction events.

## 8. Important limitations

- Loop candidates and transforms come from ground truth; place recognition is
  not implemented.
- No persistent landmarks means no classical bundle adjustment.
- The map is a keyframe point-cloud union, not a TSDF or surfel map.
- Dynamic objects, exposure changes and depth artifacts are not modeled.
- Covariance is an intuitive process-noise visualization, not a calibrated EKF.
- Processing is CPU-based and intentionally sequential inside the odometry
  worker so stage latency remains easy to understand.

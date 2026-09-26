# RGB-D SLAM learning demo

An educational extension of the official
[Open3D RGB-D odometry example](examples/python/pipelines/rgbd_odometry.py),
using the real TUM `freiburg1_xyz` sequence. This is localization plus simulated
loop closure, not a production SLAM system.

## Run

The dataset and environment are already installed in this directory:

```bash
cd /home/jacob/airobotics/3d_slam_example
source .venv/bin/activate
python slam_demo.py
```

For a clean reinstall: `./setup.sh && python download_dataset.py`.
For SSH/CI: `python slam_demo.py --no-viewer --max-frames 20 --no-realtime`.

## What to watch

- `TUM RGB-D input` window: the synchronized RGB frame and metric depth image
  actually consumed by odometry. Depth is dense, so there are no sparse
  keypoints in this algorithm.
- Colored point cloud: incremental map in the corrected world frame.
- Cyan cloud: the current depth scan placed inside the accumulated map.
- Magenta sphere and cyan frustum: exact optimized camera location and view.
- Red / orange / green / blue: noisy simulation / raw local odometry / globally
  optimized trajectory / TUM ground truth.
- Yellow wire ellipsoid: `3 sigma` positional covariance.
- XYZ axes: fixed map origin (the first ground-truth camera pose).
- Terminal and `telemetry.csv`: I/O, odometry, mapping, total latency, processing
  Hz, viewer Hz, publish-to-render latency, failures, and correction jumps.

3D controls: left-drag to orbit/tilt; `Ctrl`+left-drag or middle-drag to pan;
mouse wheel to zoom; `R` to reset the view; `H` for Open3D's built-in help.
Press `Space` in either window to pause/resume computation while the map remains
interactive. Press `Q` in the image window or `Esc` to quit. Use
`--no-image-window` for 3D only or `--no-viewer` for fully headless execution.

Every 50 processed frames, a synthetic GT loop constraint triggers Open3D
Levenberg-Marquardt pose-graph optimization. It corrects the complete green path
and rebuilds the map; orange and red are deliberately left uncorrected. Open3D
hybrid photometric/geometric RGB-D odometry supplies motion between constraints.

Read [`SLAM_GUIDE.md`](SLAM_GUIDE.md) for the algorithm, coordinate frames,
script/function map, parameters, experiments, and the difference between pose
graph optimization and bundle adjustment.

## Parameters to try

Edit `config.yaml`, then rerun:

| Parameter | Observe |
|---|---|
| `noise.*` | Faster/slower red trajectory drift |
| `loop_closure_every_n_frames` | Correction frequency and green jumps (`0` disables) |
| `covariance.*` | Ellipsoid growth and shrink after closure |
| `voxel_size_m` | Map detail versus speed |
| `depth_diff_max_m` | Odometry rejection sensitivity |
| `frame_stride` | Motion baseline versus compute cost |
| `map_every_n_frames` | Map density versus viewer latency |

Suggested experiments: disable loop closure; raise translation noise to `0.02`;
compare voxel sizes `0.02` and `0.10`; then inspect `telemetry.csv`.

```bash
# Orange local odometry and green are identical: no global correction.
python slam_demo.py --no-loop-closure

# Make corrections frequent and obvious.
python slam_demo.py --loop-closure-every 20
```

## Provenance

This directory is a sparse clone of [`isl-org/Open3D`](https://github.com/isl-org/Open3D)
at commit `b6c5e196`. The original reconstruction system remains under
`examples/python/reconstruction_system/`; the root scripts are the teaching
adapter. Open3D is MIT licensed (`LICENSE`). Dataset: TUM RGB-D benchmark,
Sturm et al., IROS 2012.

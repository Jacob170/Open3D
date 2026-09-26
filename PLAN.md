# Implementation lock

- Goal: turn the official Open3D RGB-D odometry example into a runnable TUM
  RGB-D learning demo without replacing the upstream examples.
- Required behavior: RGB/depth/ground-truth loading, drifting and corrected
  poses, 6x6 covariance, incremental 3D view, and per-stage telemetry.
- Implementation: add standalone root-level scripts and configuration; retain
  the sparse Open3D clone and its Git history as the reference implementation.
- Test: compile all Python, test dataset association, then process at least ten
  real TUM frames with `--no-viewer`.
- Risks: desktop OpenGL is unavailable in headless sessions (mitigated by
  `--no-viewer`); RGB-D odometry can fail on blurred frames (hold last motion
  and expose the failure in telemetry).

Status: implementation complete. Syntax and real-data headless execution passed;
final loop-closure and geometry checks are recorded in the session results.

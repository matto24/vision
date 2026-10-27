# Computer Vision

Exercises and the final project from a university course in computer vision. The exercises go from camera models and stereo geometry, through point-cloud processing, to 3D pose estimation and tracking. The course textbook is *Aanæs (2015), Lecture Notes on Computer Vision*.

## Contents

| Folder | Topic | What the code does |
|---|---|---|
| [`01_camera_calibration`](01_camera_calibration) | Camera calibration | Finds checkerboard corners in images, estimates the camera matrix and lens distortion, undistorts a sample image and reports the reprojection error. |
| [`02_stereo_triangulation`](02_stereo_triangulation) | Stereopsis & epipolar geometry | Builds the projection matrices of a calibrated stereo pair, then computes the optical centres, epipoles and fundamental matrix. Draws the epipolar line for a clicked point and triangulates a 3D point from a click in each image (using Plücker lines). |
| [`03_disparity_depth_maps`](03_disparity_depth_maps) | Disparity & depth | Computes disparity maps for the Tsukuba stereo pair with StereoBM / StereoSGBM, reprojects them to 3D and saves a coloured point cloud (`cloud.pcd`) and the disparity image. |
| [`04_homography`](04_homography) | Planar homography | Uses the four corners of an A4 sheet to compute a homography, warps the photo to a top-down view at 10 px/mm, and measures a PCB lying on the paper (≈147 × 208 mm). |
| [`05_pointcloud_preprocessing`](05_pointcloud_preprocessing) | Point-cloud preprocessing | Cleans up a point cloud with Open3D: crops it to a box, downsamples it with a voxel grid and removes outliers. Shows which points were removed and saves the result. |
| [`06_marker_pose_uncertainty`](06_marker_pose_uncertainty) | Marker pose uncertainty | Calibrates the camera from 100 checkerboard images and finds the pose of an ArUco marker with `solvePnP`. Then adds increasing Gaussian noise to the marker corners and plots the resulting translation error (`uncertainty.png`). |
| [`07_pointcloud_pose_estimation`](07_pointcloud_pose_estimation) | 3D pose estimation (point clouds) | Finds an object's pose in a point-cloud scene: ICP for local refinement (ex1), FPFH feature matching with RANSAC for global alignment (ex2), and both chained together on real scanned data (ex3). |
| [`08_linemod_template_matching`](08_linemod_template_matching) | 2D–3D pose with LINEMOD | `create_templates.py` renders views of a 3D model (the LINEMOD duck) from points on a sphere around it with pyrender. `linemod_template.py` matches those templates against real images with OpenCV's LINEMOD detector to get a rough pose. |
| [`09_kalman_filter`](09_kalman_filter) | Kalman filtering | Simulates a noisy projectile trajectory and compares three Kalman filters (constant position / velocity / acceleration). Includes a stretch with no measurements, all drawn live against the ground truth. |
| [`project_3d_pose_estimation`](project_3d_pose_estimation) | Final project | See below. |

### Final project: 3D pose estimation

Estimates the 6-DoF pose of a duck model (`Scenes/duck.stl`) in depth-camera point clouds. The pipeline in `Code/do_pe.py`:

1. Crops the scene, then removes planes (RANSAC) and outliers.
2. Does a global alignment with RANSAC on FPFH feature matches.
3. Refines the pose with ICP.

The scripts run this on several scenes and noise levels and compare each pose with the ground truth (`Scenes/gt_XXXX.txt`). The results go to `results/` and the plots (success rate, position/rotation error, runtime) to `graphs/`. On the noise-free scenes the error is roughly 0.3–3° and 3–3.5 mm, which is inside the course's 5° / 5 mm success threshold.

```bash
cd project_3d_pose_estimation
uv run python Code/trial_run.py   # one scene, for testing
uv run python Code/run_all.py     # all scenes and noise levels (~1 hour)
uv run python Code/visu1.py       # per-scene graphs
uv run python Code/visu2.py       # per-noise-level graphs
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12, pinned in `.python-version`. Open3D 0.19 has no wheels for newer Python versions.

```bash
uv sync
```

## Running the exercises

Paths in the scripts are relative to their own folder, so run each script from inside its folder:

```bash
cd 04_homography
uv run python homography_template.py
```

Most scripts open OpenCV or Open3D windows. Some are interactive: in `02_stereo_triangulation` you click points in the images, and `03_disparity_depth_maps` asks for parameters in the terminal. Some scripts also overwrite the output files kept in their folder, such as `cloud.pcd`, `disparity.png`, `PCB_warp.jpg` and `uncertainty.png`.

## Status

Every exercise is complete and runs. Known limitations:

- **03:** disparities are not divided by OpenCV's ×16 fixed-point factor, so depths are 16× too small. The shape of the cloud is still right.
- **07 ex3:** the alignment on real data is only rough (median error ≈3.5 mm). Feature matching is a pure-Python loop and takes several minutes.
- **09:** the optional bonus (plotting the error over time) is not done yet.
- **Project:** there is a constant ≈3 mm position offset from point-to-point ICP. RANSAC sometimes gives a flipped pose at high noise.

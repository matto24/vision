import open3d as o3d
import numpy as np
from tqdm import tqdm
import random

from ex1_pose_est_local_template import (
    show_pointclouds,
    create_kdtree,
    set_ICP_parameters,
    find_closest_points,
    estimate_transformation,
    apply_pose,
    update_result_pose_ICP,
)
from ex2_pose_est_global_template import (
    set_RANSAC_parameters,
    compute_surface_normals,
    compute_shape_features,
    find_feature_matches,
    sample_3_random_correspondences,
    validate,
    update_result_pose,
)


def load_pointclouds():
    """
    Load the real data set point clouds.
    The real data objects are stored as stl files, they need to be converted to point clouds.
    """
    mesh_obj = o3d.io.read_triangle_mesh("./datasets/parts_real/object1-global.stl")
    obj = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_obj, 2000)
    scn = o3d.io.read_point_cloud("./datasets/parts_real/scene.pcd")

    return obj, scn


def spatial_filter(input_cloud):
    # Remove points in the pointcloud that do not belong to the bust, using a Spatial Filter (passthrough)
    # Hint: You can implement this by simply cropping the pointcloud within min/max bounds
    print(
        f"Point cloud bounds: min={np.asarray(input_cloud.get_min_bound())}, max={np.asarray(input_cloud.get_max_bound())}"
    )
    min_bound = np.array([-200, -300, 0])
    max_bound = np.array([325, 365, 1371])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return input_cloud.crop(bbox)


def preprocess_pointclouds(obj, scn):
    """
    Preprocess the real data set point clouds.
    You should use some of the methods from last week.
    You should at LEAST downsample the point clouds using a voxel grid (Need to be same size).

    Link: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    """

    scn = spatial_filter(scn)
    obj = obj.voxel_down_sample(1)
    scn = scn.voxel_down_sample(1)

    return obj, scn


def RANSAC(obj, scn):
    """
    Implement the code from ex2_pose_est_global_solution.py here.
    """
    # Show starting object and scene
    show_pointclouds(obj, scn, window_name="Before global alignment")
    # Set RANSAC parameters
    it = 1000
    thressq = 0.01
    # Compute surface normals
    print("compute surface normals...")
    compute_surface_normals(obj, scn)
    # Compute shape features
    print("compute shape features...")
    obj_features, scn_features = compute_shape_features(obj, scn)
    obj_features = np.asarray(obj_features.data).T
    scn_features = np.asarray(scn_features.data).T

    print("Find feature matches...")
    corr = find_feature_matches(obj_features, scn_features)
    print("Create k-d tree...")
    tree = create_kdtree(scn)
    # Start RANSAC
    random.seed(123456789)
    inliers_best = 0
    pose_best = None
    for i in tqdm(range(it), desc="RANSAC"):
        # Sample 3 random correspondences
        corr_i = sample_3_random_correspondences(corr)
        # Estimate transformation
        T = estimate_transformation(obj, scn, corr_i)
        # Apply pose (to a copy of the object)
        obj_aligned = o3d.geometry.PointCloud(obj)
        obj_aligned = apply_pose(obj_aligned, T)
        # Validate
        inliers = validate(obj_aligned, tree, thressq)
        # Update result
        pose_best, inliers_best = update_result_pose(
            pose_best, T, inliers, inliers_best, obj_aligned
        )
    # Print pose
    print("Got the following pose:")
    print(pose_best)
    # Apply pose to the original object
    obj = apply_pose(obj, pose_best)
    # Show result
    show_pointclouds(obj, scn, window_name="After global alignment")
    return obj, scn


def ICP(obj, scn):
    """
    Implement the code from ex1_pose_est_local_solution.py here.
    """
    # Show starting object and scene
    show_pointclouds(obj, scn, window_name="Before local alignment")
    # Create a k-d tree for scene
    tree = create_kdtree(scn)
    # Set ICP parameters
    it = 1000
    thressq = 1
    # Start ICP
    pose = None
    obj_aligned = o3d.geometry.PointCloud(obj)
    for i in tqdm(range(it), desc="ICP"):
        # Find closest points
        corr = find_closest_points(obj_aligned, tree, thressq)
        # Estimate transformation
        T = estimate_transformation(obj_aligned, scn, corr)
        # Apply pose
        obj_aligned = apply_pose(obj_aligned, T)
        # Update result
        pose = update_result_pose_ICP(pose, T)
    # Print pose
    print("Got the following pose:")
    print(pose)
    # Apply pose to the original object
    obj = apply_pose(obj, pose)
    # Show result object and scene
    show_pointclouds(obj, scn, window_name="After local alignment")
    return obj, scn


def main():
    print("Loading point clouds...")
    obj, scn = load_pointclouds()
    print("Preprocessing point clouds...")
    obj, scn = preprocess_pointclouds(obj, scn)
    print("RANSAC...")
    obj, scn = RANSAC(obj, scn)
    print("ICP...")
    obj, scn = ICP(obj, scn)


if __name__ == "__main__":
    main()

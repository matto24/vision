import open3d as o3d
import numpy as np
from tqdm import tqdm
import random

from ex1_pose_est_local_template import show_pointclouds, create_kdtree, set_ICP_parameters, find_closest_points, estimate_transformation, apply_pose, update_result_pose_ICP
from ex2_pose_est_global_template import set_RANSAC_parameters, compute_surface_normals, compute_shape_features, find_feature_matches, sample_3_random_correspondences, validate, update_result_pose

def load_pointclouds():
    """
    Load the real data set point clouds.
    The real data objects are stored as stl files, they need to be converted to point clouds.
    """
    return obj, scn

def preprocess_pointclouds(obj, scn):
    """
    Preprocess the real data set point clouds.
    You should use some of the methods from last week.
    You should at LEAST downsample the point clouds using a voxel grid (Need to be same size).

    Link: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    """
    return obj, scn

def RANSAC(obj, scn):
    """
    Implement the code from ex2_pose_est_global_solution.py here.
    """
    # Show starting object and scene
    show_pointclouds(obj, scn, window_name='Before global alignment')
    # Set RANSAC parameters
    it, thressq = set_RANSAC_parameters() # Probably a good idea to change this and assign the values directly here
    # Compute surface normals
    compute_surface_normals(obj, scn)
    # Compute shape features
    obj_features, scn_features = compute_shape_features(obj, scn)
    obj_features = np.asarray(obj_features.data).T
    scn_features = np.asarray(scn_features.data).T
    corr = find_feature_matches(obj_features, scn_features)
    tree = create_kdtree(scn)
    # Start RANSAC
    random.seed(123456789)
    inliers_best = 0
    pose_best = None
    for i in tqdm(range(it), desc='RANSAC'):   
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
        pose_best, inliers_best = update_result_pose(pose_best, T, inliers, inliers_best, obj_aligned)
    # Print pose
    print('Got the following pose:')
    print(pose_best)
    # Apply pose to the original object
    obj = apply_pose(obj, pose_best)
    # Show result
    show_pointclouds(obj, scn, window_name='After global alignment')
    return obj, scn

def ICP(obj, scn):
    """
    Implement the code from ex1_pose_est_local_solution.py here.
    """
    # Show starting object and scene
    show_pointclouds(obj, scn, window_name='Before local alignment')
    # Create a k-d tree for scene
    tree = create_kdtree(scn)
    # Set ICP parameters
    it, thressq = set_ICP_parameters() # Probably a good idea to change this and assign the values directly here
    # Start ICP
    pose = None
    obj_aligned = o3d.geometry.PointCloud(obj)
    for i in tqdm(range(it), desc='ICP'):
        # Find closest points
        corr = find_closest_points(obj_aligned, tree, thressq)
        # Estimate transformation
        T = estimate_transformation(obj_aligned, scn, corr)
        # Apply pose
        obj_aligned = apply_pose(obj_aligned, T)
        # Update result
        pose = update_result_pose_ICP(pose, T)
    # Print pose
    print('Got the following pose:')
    print(pose)
    # Apply pose to the original object
    obj = apply_pose(obj, pose)
    # Show result object and scene
    show_pointclouds(obj, scn, window_name='After local alignment')
    return obj, scn

def main():
    obj, scn = load_pointclouds()
    obj, scn = preprocess_pointclouds(obj, scn)
    obj, scn = RANSAC(obj, scn)
    obj, scn = ICP(obj, scn)
    


if __name__ == '__main__':
    main()
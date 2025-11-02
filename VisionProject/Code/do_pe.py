import open3d as o3d
import copy
import numpy as np
from tqdm import tqdm
import random


def show_pointclouds(obj, scn, window_name):
    """
    Display the object and scene point clouds in a visualizer window.

    Hint: Use `o3d.visualization.draw_geometries` to visualize.
    Link: https://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries.html
    """

    o3d.visualization.draw_geometries([obj, scn], window_name=window_name)


def spatial_filter(input_cloud):
    # Remove points in the pointcloud that do not belong to the bust, using a Spatial Filter (passthrough)
    # Hint: You can implement this by simply cropping the pointcloud within min/max bounds
    print(
        f"Point cloud bounds: min={np.asarray(input_cloud.get_min_bound())}, max={np.asarray(input_cloud.get_max_bound())}"
    )
    min_bound = np.array([-0.6, -0.5, 0.9])
    max_bound = np.array([0.6, 0.5, 1.5])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    input_cloud = input_cloud.crop(bbox)
    return input_cloud


def remove_plane(input_cloud):
    # remove plane
    plane_model, inliers = input_cloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )

    # remove inliers
    input_cloud = input_cloud.select_by_index(inliers, invert=True)

    plane_model, inliers = input_cloud.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )

    # remove inliers
    input_cloud = input_cloud.select_by_index(inliers, invert=True)

    return input_cloud


def display_removal(preserved_points, removed_points):
    removed_points.paint_uniform_color([1, 0, 0])  # Show removed points in red
    preserved_points.paint_uniform_color(
        [0.8, 0.8, 0.8]
    )  # Show preserved points in gray
    o3d.visualization.draw_geometries([removed_points, preserved_points])


def preprocess_pointclouds(obj, scn):
    """
    Preprocess the real data set point clouds.
    You should use some of the methods from last week.
    You should at LEAST downsample the point clouds using a voxel grid (Need to be same size).

    Link: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    """

    scn = spatial_filter(scn)
    scn = remove_plane(scn)
    scn = outlier_removal(scn)
    # show_pointclouds(obj, scn, "filtered")

    return obj, scn


def outlier_removal(input_cloud):
    # Remove outliers from the pointcloud using a statistical outlier removal filter
    cloud_filtered, ind = input_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )

    # display_removal(cloud_filtered, input_cloud.select_by_index(ind, invert=True))

    return cloud_filtered


def compute_surface_normals(obj, scn):
    """
    Compute surface normals for both object and scene point clouds.

    Expected input: Object and scene point clouds.
    Expected output: None (normals are estimated in place, and saved in the point cloud object).

    Hint: Use `estimate_normals` with `KDTreeSearchParamKNN`.
    Link: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
    (Under Vertex normal estimation is an example, but I recommend using search_param=o3d.geometry.KDTreeSearchParamKNN instead of HybridParam).
    """
    obj.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    scn.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))

    return


def find_feature_matches(obj_features, scn_features):
    """
    Find feature correspondences between the object and scene features.

    Expected input: Object and scene features.
    Expected output: Correspondences as a Vector2iVector. Each element is a pair of indices: (obj_feats_index, scene_feats_index).

    Hint: Loop over the object features, compute squared distances, and find the minimum.
    """

    corr = []

    for i in range(len(obj_features)):
        smallest_dist = float("inf")
        smallest_dist_idx = 0
        for j in range(len(scn_features)):
            dist = np.linalg.norm(obj_features[i] - scn_features[j])
            if dist < smallest_dist:
                smallest_dist = dist
                smallest_dist_idx = j

        corr.append([i, smallest_dist_idx])

    return o3d.utility.Vector2iVector(corr)


def create_kdtree(scn):
    """
    Create a k-d tree from the scene point cloud for efficient nearest neighbor search.

    Expected input: Scene point cloud.
    Expected output: KDTreeFlann object.

    Hint: Use `o3d.geometry.KDTreeFlann`.
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html
    """
    tree = o3d.geometry.KDTreeFlann(scn)
    return tree


def compute_shape_features(obj, scn):
    """
    Compute shape features for both object and scene point clouds. The template assumes FPFH features.

    Expected input: Object and scene point clouds.
    Expected output: FPFH features for object and scene.

    Hint: Use `compute_fpfh_feature` with a `KDTreeSearchParamRadius`.
    Link: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.compute_fpfh_feature.html
    """
    obj_features = o3d.pipelines.registration.compute_fpfh_feature(
        obj, o3d.geometry.KDTreeSearchParamRadius(0.01)
    )
    scn_features = o3d.pipelines.registration.compute_fpfh_feature(
        scn, o3d.geometry.KDTreeSearchParamRadius(0.01)
    )
    return obj_features, scn_features


def sample_3_random_correspondences(corr):
    """
    Randomly sample 3 correspondences from the full correspondence set.

    Expected input: Correspondences as a Vector2iVector.
    Expected output: A set (Vector2iVector) of 3 random correspondences.

    Hint: Use `random.choices()` to sample 3 correspondences.
    Link: https://docs.python.org/3/library/random.html#random.choices
    """

    random_list = random.choices(corr, k=3)
    random_corr = o3d.utility.Vector2iVector(random_list)
    return random_corr


def estimate_transformation(obj, scn, corr):
    """
    Estimate the transformation matrix from the correspondences between object and scene.

    Expected input: Object point cloud, scene point cloud, and correspondences.
    Expected output: Transformation matrix.

    Hint: Use `TransformationEstimationPointToPoint().compute_transformation`. You need to create an instance of the class first.
    Link: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.TransformationEstimationPointToPoint.html
    """

    T = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
        obj, scn, corr
    )
    return T


def apply_pose(obj, T):
    """
    Apply the transformation matrix to the object point cloud.

    Expected input: Object point cloud and transformation matrix.
    Expected output: Transformed object point cloud.

    Hint: This is a very short function.
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.transform
    """
    obj.transform(T)
    return obj


def validate(obj_aligned, tree, thressq):
    """
    Validate the transformation by counting the number of inliers based on the distance threshold.

    Expected input: Object point cloud, KDTreeFlann object, and squared distance threshold.
    Expected output: Number of inliers (int).

    Hint: You need to go through each point of the aligned object and find the closest point in the scene.
    Hint: Use `tree.search_knn_vector_3d` to find the closest point and then apply the distance threshold.
          Then count the number of inliers (points where the closest point is within the threshold).
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html
    """

    inliers = 0
    points = obj_aligned.points

    for i in range(len(points)):
        [k, idx, dist] = tree.search_knn_vector_3d(points[i], 1)
        if dist[0] < thressq:
            inliers += 1

    return inliers


def update_result_pose(pose_best, T, inliers, inliers_best, obj):
    """
    Update the best pose based on the number of inliers found.

    Expected input: Best pose, current pose (T), number of inliers, best inliers, and object point cloud.
    Expected output: Updated pose and inlier count.

    Hint: Compare the current number of inliers to the best so far, and update pose_best and inliers_best they are greater.
    Hint: RANSAC, unlike ICP, does not accumulate transformations. Instead, you need to store the best transformation so far.
    Hint: The object point cloud "obj" is only here, so you can print something like this on success:
          print(f'Got a new model with {inliers}/{len(obj.points)} inliers!')
    """

    if inliers > inliers_best:
        pose_best = T
        inliers_best = inliers
        print(f"Got a new model with {inliers}/{len(obj.points)} inliers!")

    return pose_best, inliers_best


def RANSAC(obj, scn):
    """
    Implement the code from ex2_pose_est_global_solution.py here.
    """
    # Show starting object and scene
    # show_pointclouds(obj, scn, window_name="Before global alignment")
    # Set RANSAC parameters

    ransac_obj = copy.deepcopy(obj)
    ransac_scn = copy.deepcopy(scn)

    ransac_obj = ransac_obj.voxel_down_sample(0.005)
    ransac_scn = ransac_scn.voxel_down_sample(0.005)

    it = 3000
    thressq = 0.0001
    # Compute surface normals
    print("compute surface normals...")
    compute_surface_normals(ransac_obj, ransac_scn)
    # Compute shape features
    print("compute shape features...")
    obj_features, scn_features = compute_shape_features(ransac_obj, ransac_scn)
    obj_features = np.asarray(obj_features.data).T
    scn_features = np.asarray(scn_features.data).T

    print("Find feature matches...")
    corr = find_feature_matches(obj_features, scn_features)
    print("Create k-d tree...")
    tree = create_kdtree(ransac_scn)
    # Start RANSAC
    random.seed(123456789)
    inliers_best = 0
    pose_best = None
    for i in tqdm(range(it), desc="RANSAC"):
        # Sample 3 random correspondences
        corr_i = sample_3_random_correspondences(corr)
        # Estimate transformation
        T = estimate_transformation(ransac_obj, ransac_scn, corr_i)
        # Apply pose (to a copy of the object)
        obj_aligned = o3d.geometry.PointCloud(ransac_obj)
        obj_aligned = apply_pose(obj_aligned, T)
        # Validate
        inliers = validate(obj_aligned, tree, thressq)
        # Update result
        pose_best, inliers_best = update_result_pose(
            pose_best, T, inliers, inliers_best, obj_aligned
        )
    # Print pose
    # print("Got the following pose:")
    # print(pose_best)
    # Apply pose to the original object
    # ransac_obj = apply_pose(obj, pose_best)
    # Show result
    # show_pointclouds(obj, scn, window_name="After global alignment")
    return pose_best


def find_closest_points(obj_aligned, tree, thressq):
    """
    Find closest points between the object and the scene point clouds using the k-d tree.

    Expected input: Object point cloud, KDTreeFlann object, and squared distance threshold.
    Expected output: Correspondences as a Vector2iVector. Each element is a pair of indices: (object_index, scene_index).

    Hint: You need to go through each point of the aligned object and find the closest point in the scene.
    Hint: Use `tree.search_knn_vector_3d` to find the closest point and then apply the distance threshold (append the correspondence to output if within).
    Link: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeFlann.html#open3d.geometry.KDTreeFlann.search_knn_vector_3d
    """
    corr = []

    # iterate through each point in obj_aligned
    points = obj_aligned.points
    for i in range(len(points)):
        # find nearest point
        [k, idx, dist] = tree.search_knn_vector_3d(points[i], 1)
        if dist[0] < thressq:
            corr.append([i, idx[0]])

    return o3d.utility.Vector2iVector(corr)


def update_result_pose_ICP(pose, T):
    """
    Update the overall transformation (pose) by multiplying it with the current transformation.

    Expected input: Current pose (None on first iteration) and current transformation.
    Expected output: Updated pose.

    Hint: Very short if/else statement. Apply matrix multiplication (`@`) to update the pose.
    """

    if pose is None:
        pose = T
    else:
        pose = pose @ T
    return pose


def ICP(obj, scn):
    """
    Implement the code from ex1_pose_est_local_solution.py here.
    """
    # Show starting object and scene
    # show_pointclouds(obj, scn, window_name="Before local alignment")
    # Create a k-d tree for scene
    obj_icp = copy.deepcopy(obj)
    scn_icp = copy.deepcopy(scn)

    obj_icp = obj_icp.voxel_down_sample(0.003)
    scn_icp = scn_icp.voxel_down_sample(0.003)

    tree = create_kdtree(scn_icp)
    # Set ICP parameters
    it = 2000
    thressq = 0.0001
    # Start ICP
    pose = None
    obj_aligned = o3d.geometry.PointCloud(obj_icp)
    for i in tqdm(range(it), desc="ICP"):
        # Find closest points
        corr = find_closest_points(obj_aligned, tree, thressq)
        # Estimate transformation
        T = estimate_transformation(obj_aligned, scn_icp, corr)
        # Apply pose
        obj_aligned = apply_pose(obj_aligned, T)
        # Update result
        pose = update_result_pose_ICP(pose, T)
    # Print pose
    # print("Got the following pose:")
    # print(pose)
    # Apply pose to the original object
    obj = apply_pose(obj, pose)
    # Show result object and scene
    # show_pointclouds(obj, scn, window_name="After local alignment")
    return pose


def do_pose_estimation(scene_pointcloud, object_pointcloud):
    print("preprocess_pointclouds")
    obj, scn = preprocess_pointclouds(object_pointcloud, scene_pointcloud)
    print("RANSAC")

    pose_ran = RANSAC(obj, scn)
    print("ICP...")
    obj_to_icp = copy.deepcopy(obj)
    apply_pose(obj_to_icp, pose_ran)
    pose_icp = ICP(obj_to_icp, scn)

    pose = pose_icp @ pose_ran
    return pose

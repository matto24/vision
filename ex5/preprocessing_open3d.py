import open3d as o3d
import numpy as np
from tqdm import tqdm


# This function just displays the effect of one of the functions visually, feel free to ignore or remove it.
def display_removal(preserved_points, removed_points):
    removed_points.paint_uniform_color([1, 0, 0])  # Show removed points in red
    preserved_points.paint_uniform_color(
        [0.8, 0.8, 0.8]
    )  # Show preserved points in gray
    o3d.visualization.draw_geometries([removed_points, preserved_points])


def voxel_grid(input_cloud):
    voxel_size = 0.1  # voxel size in meters
    return input_cloud.voxel_down_sample(voxel_size)
    # Downsample the pointcloud using a Voxel Grid Filter.


def outlier_removal(input_cloud):
    # Remove outliers from the pointcloud using a statistical outlier removal filter
    cloud_filtered, ind = input_cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    # Hint: The filter will return an index variable "ind", that can be used to select the PC, like so:
    display_removal(cloud_filtered, input_cloud.select_by_index(ind, invert=True))
    return cloud_filtered


def spatial_filter(input_cloud):
    # Remove points in the pointcloud that do not belong to the bust, using a Spatial Filter (passthrough)
    # Hint: You can implement this by simply cropping the pointcloud within min/max bounds
    print(
        f"Point cloud bounds: min={np.asarray(input_cloud.get_min_bound())}, max={np.asarray(input_cloud.get_max_bound())}"
    )
    min_bound = np.array([-50, -80, 275])
    max_bound = np.array([132, 120, 300])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return input_cloud.crop(bbox)


def main():
    # Load pointcloud (unfiltered)
    cloud = o3d.io.read_point_cloud("./ex5/cloud.pcd")

    # Show
    o3d.visualization.draw_geometries(
        [cloud], window_name="Pointcloud before filtering"
    )

    # print("PointCloud before filtering: {} data points".format(cloud.points.shape[0]))
    print("PointCloud before filtering: {} data points".format(len(cloud.points)))

    cloud_filtered = spatial_filter(cloud)
    cloud_filtered = voxel_grid(cloud_filtered)
    cloud_filtered = outlier_removal(cloud_filtered)

    # print("PointCloud after filtering: {} data points".format(cloud_filtered.points.shape[0]))
    print(
        "PointCloud after filtering: {} data points".format(len(cloud_filtered.points))
    )

    o3d.io.write_point_cloud("cloud_filtered_py.pcd", cloud_filtered)

    # Show
    o3d.visualization.draw_geometries(
        [cloud_filtered], window_name="Pointcloud after filtering"
    )


if __name__ == "__main__":
    main()

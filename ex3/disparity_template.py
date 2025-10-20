import cv2
import numpy as np
import open3d as o3d


# Save the point cloud
def save_point_cloud(filename, points, colors, max_z):
    pcd = o3d.geometry.PointCloud()
    valid_points = []
    valid_colors = []

    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            xyz = points[i, j]
            bgr = colors[i, j]

            # Only include points within the specified max_z range
            if abs(xyz[2]) < max_z:
                valid_points.append([xyz[0], xyz[1], xyz[2]])
                valid_colors.append([bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0])

    pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(valid_colors))

    o3d.io.write_point_cloud(filename, pcd)


# Define the Q matrix
def define_Q(img_width, img_height):
    # Define the (artificial) 4x4 perspective transformation matrix Q
    # Follow the structure from the slides
    cx = img_width / 2
    cy = img_height / 2
    cmx = cx
    Tx = -100
    f = 500
    Q = np.matrix(
        f"1 0 0 {-cx}; 0 1 0 {-cy}; 0 0 0 {f}; 0 0 {-1 / Tx} {(cx - cmx) / Tx}"
    )
    return Q


# Block Matching (BM) disparity calculation
def disparity_BM(imgL, imgR, nDisparities, blockSize):
    # Create a StereoBM object
    # Compute the disparity map using the StereoBM object
    stereoBM = cv2.StereoBM.create(nDisparities, blockSize)
    disparity = stereoBM.compute(imgL, imgR)
    return disparity


# Semi-Global Block Matching (SGBM) disparity calculation
def disparity_SGBM(imgL, imgR, nDisparities, blockSize):
    # Create a StereoSGBM object
    stereoSGBM = cv2.StereoSGBM.create(0, nDisparities, blockSize)
    disparity = stereoSGBM.compute(imgL, imgR)
    # Compute the disparity map using the StereoSGBM object
    return disparity


# Normalize the disparity map for visualization
def norm_disparity(disp):
    # Normalize the disparity map to unsigned 8-bit integers between 0 and 255 for visualization

    disp = cv2.normalize(disp, disp, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    disp = np.uint8(disp)
    # hint: a normalize function from OpenCV
    return disp


# Reproject to 3D using the disparity map and Q matrix
def reproject_3D(disp, Q):
    # Reproject the disparity map to 3D using the perspective transformation matrix Q
    # hint: use the specific function from OpenCV
    points = cv2.reprojectImageTo3D(disp, Q)
    return points


# Main function
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 1:
        print(f"{sys.argv[0]}")
        sys.exit(-1)

    # Load images
    imgL = cv2.imread("tsukuba_l.png")
    imgR = cv2.imread("tsukuba_r.png")

    if imgL is None or imgR is None:
        print("Error loading the images")
        sys.exit(-1)

    # Convert to grayscale
    colors = imgL
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    disp = None
    print("Try different stereo parameters")

    while True:
        algo = input("\nChoose algorithm (bm/sgbm): ").strip().lower()
        if algo not in ["bm", "sgbm"]:
            print("Wrong input")
            continue

        nDisparities = int(input("Choose nDisparities: ").strip())
        blockSize = int(input("Choose SADWindowSize: ").strip())

        # Choose the algorithm
        if algo == "bm":
            disp = disparity_BM(imgL_gray, imgR_gray, nDisparities, blockSize)
        elif algo == "sgbm":
            disp = disparity_SGBM(imgL_gray, imgR_gray, nDisparities, blockSize)

        disp_norm = norm_disparity(disp)
        print(
            "Press ESC to save the disparity map and continue to computing the point cloud"
        )
        print("Press any other key to try block matching again")
        cv2.imshow("Stereo", disp_norm)  # type: ignore

        # Press esc to save the disparity map, and exit this loop. Otherwise, try block matching again, with new parameters
        if cv2.waitKey(0) == 27:  # 27 is the 'ESC' key
            cv2.imwrite("disparity.png", disp_norm)  # type: ignore
            break

    # Define the Q matrix
    qMat = define_Q(imgL_gray.shape[1], imgL_gray.shape[0])

    # Reproject to 3D
    points = reproject_3D(disp, qMat)

    # Save the point cloud
    z_threshold = 500
    save_point_cloud("cloud.pcd", points, colors, z_threshold)

    print("Done")

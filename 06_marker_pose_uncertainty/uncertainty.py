import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def add_noise(corners, std_dev):
    """
    Add noise to the image coordinates of the 4 corners of the marker.
    The noise should be sampled from a Gaussian distribution with mean 0 and standard deviation std_dev
    And applied to both X and Y direction
    """
    noisy_corners = corners.copy()

    noise = np.random.normal(loc=0, scale=std_dev, size=(4, 2))
    noisy_corners += noise

    return noisy_corners


def calculate_error(errors):
    """Calculate a scalar error value from the error vector.
    Some examples could be:
     - The largest L2 norm (Euclidean distance) of the error of any trial
     - The L2 norm of the mean of the errors of all trials
     - The mean of the L2 norm of the errors of all trials
     - The L2 norm of the std_dev of the errors of all trials
     - The std_dev of the L2 norm of the errors of all trials
     - The largest axis of the uncertainty ellipsoid
    """

    norms = np.linalg.norm(errors, axis=1)
    scalar_error = np.mean(norms)

    return scalar_error


def plot(error, std_dev_step):
    """Plot the position error as a function of the std_dev of noise"""

    # --------------------------------------
    # Insert your code here
    # --------------------------------------
    # I gave you few lines below to work with, but change it to your needs

    plt.figure()
    plt.title(
        "mean of the error distance as a function of the standard deviation of noise"
    )
    plt.xlabel("Standard deviation of noise [pixels]")
    plt.ylabel("Mean of the error distance [mm]")

    error = np.array(error) * 1000  # Transform the error to millimetres

    # Plot the mean of the error as a function of the std_dev of noise
    plt.plot(np.arange(len(error)) * std_dev_step, error)
    plt.savefig("uncertainty.png")
    plt.show()


def estimate_pose(corners, square_size, cam_intr, dist_coeffs):
    """Estimate the pose of the ArUco marker given the image coordinates of the 4 corners"""
    # rvec, tvec, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, square_size, cam_intr, dist_coeffs)

    # Define the 3D object points corresponding to the corners of the ArUco marker
    object_points = np.array(
        [
            [-square_size / 2, square_size / 2, 0],
            [square_size / 2, square_size / 2, 0],
            [square_size / 2, -square_size / 2, 0],
            [-square_size / 2, -square_size / 2, 0],
        ],
        dtype=np.float32,
    )

    # Use solvePnP to estimate the pose
    _, rvec, tvec = cv2.solvePnP(object_points, corners, cam_intr, dist_coeffs)

    return rvec, tvec, object_points


def camera_calibration(dirpath, square_size, width, height, visualize=False):
    """Perform camera calibration using the images in the given directory path."""
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Get the images in the directory in alphanumeric order
    images = sorted(os.listdir(dirpath), key=lambda x: int(os.path.splitext(x)[0][5:]))

    with tqdm(
        total=len(images)
    ) as pbar:  # Create a tqdm progress bar for iterating through images
        for fname in images:
            img = cv2.imread(os.path.join(dirpath, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                255 - gray, (width, height), flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                if visualize:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                    cv2.imshow("img", img)
                    cv2.waitKey(0)
            # else:
            #     print("No corners found in image: ", fname)
            pbar.update(1)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_USE_LU
    )
    # Note: The flags for calibrateCamera and findChessboardCorners vastly improve execution time, with only a small drop in accuracy

    print("Camera calibration complete!\nCamera matrix: ", mtx)
    print("Distortion coefficients: ", dist)
    return [ret, mtx, dist, rvecs, tvecs]


def main():
    parser = argparse.ArgumentParser(
        description="Investigate uncertainty on pose estimation from noise"
    )
    # Arguments pertaining to the camera calibration (Square size is also used in the pose estimation)
    parser.add_argument(
        "-c",
        "--calib_dir",
        type=str,
        default="calibration_images/",
        help="Path to folder containing checkerboard images for calibration",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=10, help="Width of checkerboard (default=10)"
    )
    parser.add_argument(
        "-t", "--height", type=int, default=7, help="Height of checkerboard (default=7)"
    )
    parser.add_argument(
        "-q",
        "--sq_size_calib",
        type=float,
        default=0.036666667,
        help="Length of one edge (in metres) of the calibration checkerboard",
    )
    parser.add_argument(
        "-m",
        "--select_marker",
        type=int,
        default=0,
        help="Select by index which marker to test (default=0)",
    )
    parser.add_argument(
        "-vc",
        "--visualize_calib",
        type=str,
        default="False",
        help="To visualize each checkerboard image",
    )
    # Arguments pertaining to the pose estimation
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="object_images/image0001.png",
        help="Path to image of an aruco marker we want to estimate the pose of",
    )
    parser.add_argument(
        "-u",
        "--std_dev_max",
        type=float,
        default=1.5,
        help="Maximum standard deviation of noise to add to the image coordinates of the 4 corners of the marker",
    )
    parser.add_argument(
        "-r",
        "--std_dev_step",
        type=float,
        default=0.02,
        help="Step size of standard deviation of noise to add to the image coordinates of the 4 corners of the marker",
    )
    parser.add_argument(
        "-n",
        "--num_repeat",
        type=int,
        default=10000,
        help="Number of times to repeat the experiment at each noise level",
    )
    parser.add_argument(
        "-d",
        "--dictionary",
        type=str,
        default="DICT_5X5_100",
        help="Name of the aruco dictionary we want to use",
    )
    parser.add_argument(
        "-s",
        "--square_size",
        type=float,
        default=0.01125,
        help="Length of one edge (in metres)",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        type=str,
        default="False",
        help="To visualize the image with the detected marker and the estimated pose",
    )
    args = parser.parse_args()

    # Load the arguments for camera calibration
    calib_dir = getattr(args, "calib_dir")  # Path to the calibration images
    sq_size_calib = getattr(
        args, "sq_size_calib"
    )  # Length of one edge of the checkerboard (in metres)
    width = getattr(args, "width")  # Width of the checkerboard
    height = getattr(args, "height")  # Height of the checkerboard
    visu_calib = (
        getattr(args, "visualize_calib") == "True"
    )  # To visualize each checkerboard image

    # Load the arguments for the pose estimation
    std_dev_max = getattr(
        args, "std_dev_max"
    )  # Maximum standard deviation of noise to add to the image coordinates
    std_dev_step = getattr(
        args, "std_dev_step"
    )  # Step size of standard deviation of noise to add to the image coordinates
    num_repeat = getattr(
        args, "num_repeat"
    )  # Number of times to repeat the experiment at each noise level
    square_size = getattr(
        args, "square_size"
    )  # Length of one edge of the ArUco marker (in metres)
    marker_indx = getattr(args, "select_marker")  # Index of the marker to test
    visu = (
        getattr(args, "visualize") == "True"
    )  # To visualize the image with the detected marker and the estimated pose

    # ---------------------------------------------------------------------------------------------------------------
    # If the above lines of code are confusing to you:
    # It is simply a way to pass arguments to the python script from the command line (terminal)
    # For example, if you can run the script with the following arguments to change the test image:
    #   python3 uncertainty.py -i object_images/image0002.png
    #
    # You can also simply change the default values of the arguments above, and run the script without any arguments
    # ---------------------------------------------------------------------------------------------------------------

    # Calibrate the camera
    ret, cam_intr, dist_coeffs, rvecs, tvecs = camera_calibration(
        calib_dir, sq_size_calib, width, height, visu_calib
    )

    # Load the image from argument
    image = cv2.imread(getattr(args, "image"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create ArUco marker detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, getattr(args, "dictionary"))
    )  # Load dictionary from args
    aruco_params = cv2.aruco.DetectorParameters()  # Marker detection parameters

    # Detect the ArUco marker in the image and extract the 4 corners
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    # Visualize the image with the detected marker and the estimated pose
    if visu:
        img_visu = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        cv2.imshow("Detected ArUco markers", img_visu)
        cv2.waitKey(0)
    print("Number of ArUco markers detected: ", len(corners))
    print("Processing marker: ", ids[marker_indx])

    # Estimate the pose of the marker (Ground truth)
    gt_rvec, gt_tvec, _ = estimate_pose(
        corners[marker_indx], square_size, cam_intr, dist_coeffs
    )
    # print("Ground truth rotation vector: \n", gt_rvec)
    # print("Ground truth translation vector: \n", gt_tvec)

    # Visualize the image with the estimated pose
    if visu:
        img_visu = cv2.drawFrameAxes(
            image.copy(), cam_intr, dist_coeffs, gt_rvec, gt_tvec, 0.02
        )
        cv2.imshow("Ground truth pose of marker", img_visu)
        cv2.waitKey(0)

    # Create a variable to store the error for each level of noise
    noise_level_errors = []

    print("Calculating error for each noise level...")
    with tqdm(
        total=std_dev_max / std_dev_step
    ) as pbar:  # Create a tqdm progress bar for iterating through images
        # Perform the experiment on different levels of noise variance
        for i, std_dev in enumerate(np.arange(0, std_dev_max, std_dev_step)):
            # Create a variable to store the error for each level of noise
            errors = []

            # Repeat the experiment at each noise level N times (num_repeat)
            for j in range(num_repeat):
                # Add noise to the image coordinates of the 4 corners of the marker
                noisy_corners = add_noise(corners[marker_indx], std_dev)

                # Estimate the pose of the marker
                est_rvec, est_tvec, _ = estimate_pose(
                    noisy_corners, square_size, cam_intr, dist_coeffs
                )

                # Calculate the error between the estimated pose and the ground truth pose
                error = est_tvec - gt_tvec
                errors.append(error)

            # Calculate a scalar value from the error
            scalar_error = calculate_error(errors)

            # Add the scalar error to the list of errors
            noise_level_errors.append(scalar_error)
            pbar.update(1)

    # Plot the mean and standard deviation of the error as a function of the std_dev of noise
    plot(noise_level_errors, std_dev_step)


if __name__ == "__main__":
    main()

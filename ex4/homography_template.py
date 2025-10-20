import cv2
import numpy as np
from mouseclick import getMouseClick

scale = 10


def computeHomography(corners_a4_image):
    # ----------------- YOUR CODE HERE ----------------------
    # Define the paper plane by defining 4 corner points in the plane (Using the A4 paper size)
    # Compute the homography matrix using
    # 4 corner points in the image plane as srcPoints
    # 4 corner points in the A4 plane as dstPoints
    # You should use cv2.findHomography(srcPoints, dstPoints)
    # --------------------------------------------------------

    A4_size = [210, 297]
    dstPoints = np.array(
        [
            [0, 0],
            [A4_size[0] * scale, 0],
            [A4_size[0] * scale, A4_size[1] * scale],
            [0, A4_size[1] * scale],
        ]
    )
    corners_a4_image = np.array(corners_a4_image)

    H = cv2.findHomography(corners_a4_image, dstPoints)
    return H[0]


def warpImage(img, H):
    A4_size = 210, 297
    img_warp = cv2.warpPerspective(img, H, (A4_size[0] * scale, A4_size[1] * scale))
    return img_warp


def computeWidthHeight(corners_pcb_original, H):
    # ----------------- YOUR CODE HERE ----------------------
    # Transform each point using the homography matrix
    # Compute the width and height of the PCB in the warped image
    # -------------------------------------------------------

    # make the corners homogeneous
    corners = [np.hstack([i, 1]) for i in corners_pcb_original]
    print("xy, corners", corners)

    # Transformed points
    # Also called warped points
    points_pcb_plan = np.dot(H, np.array(corners).T)
    print("Points in PCB PLAN", points_pcb_plan)

    points_pcb_plan = points_pcb_plan.T

    # normalize the homogeneous coordinates
    normalized_coordinates = points_pcb_plan[:, :2] / points_pcb_plan[:, 2, np.newaxis]
    width = (
        np.linalg.norm(normalized_coordinates[1] - normalized_coordinates[0]) / scale
    )
    height = (
        np.linalg.norm(normalized_coordinates[1] - normalized_coordinates[2]) / scale
    )
    return width, height


def main():
    # Read the image
    img = cv2.imread("./PCB.jpg")

    corners_a4_image = [(736, 2540), (4288, 2288), (2984, 1080), (1392, 1136)]
    corners_pcb_original = [(1168, 2012), (3292, 2116), (3096, 1224), (1852, 1208)]

    # corners_a4_image = []
    # corners_pcb_original = []
    # Get the 4 corners of the A4 paper in the image by clicking on them
    # for i in range(4):
    #     corners_a4_image.append(getMouseClick("Click on A4 corners", img))
    # print("Clicked A4 corners in the image: ", corners_a4_image)

    # Visualize corners in the original image
    for corner in corners_a4_image:
        cv2.circle(img, corner, 50, (255, 255, 255), 10)

    # for i in range(4):
    #     corners_pcb_original.append(getMouseClick("Click on PCB corners", img))

    # print("Clicked PCB corners in the image: ", corners_pcb_original)

    # Visualize PCB corners
    for i, corner in enumerate(corners_pcb_original):
        cv2.circle(
            img, (corner[0], corner[1]), 50, ((i % 2) * 255, 0, (i // 2) * 255), 10
        )

    # Scale down image for visualization
    img_show = cv2.pyrDown(cv2.pyrDown(img))

    # Show the original image
    cv2.imshow("Original", img_show)
    cv2.waitKey()

    # Compute the homography matrix
    H = computeHomography(corners_a4_image)
    print("H:\n", H)

    # Warp the image using the homography
    img_warp = warpImage(img, H)

    # Save the warped image
    cv2.imwrite("./PCB_warp.jpg", img_warp)

    # Visualize the warped image
    img_warp_show = cv2.pyrDown(cv2.pyrDown(img_warp))
    cv2.imshow("Warp", img_warp_show)
    cv2.waitKey()

    # Compute the width and height after warping
    width, height = computeWidthHeight(corners_pcb_original, H)

    print(f"Width (in mm): {width}")
    print(f"Height (in mm): {height}")

    # Show final images for reference
    cv2.imshow("Original", img_show)
    cv2.imshow("Warp", img_warp_show)
    cv2.waitKey()


if __name__ == "__main__":
    main()

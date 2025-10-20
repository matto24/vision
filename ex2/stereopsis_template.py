import cv2
import numpy as np
import sys
from methods import loadImagesAndCalibration, getMouseClick


def constructProjectionMat(cam):
    KA = cam.intrinsic
    H = cam.transformation
    # add a column of zeros to KA
    KA = np.hstack((KA, np.zeros((3, 1))))

    P = np.matmul(KA, H)
    # ------------------------------------
    return P


def splitPp(proj):
    # ------YOUR CODE GOES HERE-----------
    Pp = [None, None]
    # Split the projection matrix into 2 matrices
    # Pp[0] should be a 3x3 matrix
    Pp[0] = proj[:3, :3]
    Pp[1] = proj[:3, 3]
    # Pp[1] should be a 3x1 matrix
    # ------------------------------------
    return Pp


#
def computeOpticalCenter(Pp):
    # Compute the optical center of the camera
    # Remember return the result in homogeneous coordinates (Append a 1)
    C = np.matmul(-np.linalg.inv(Pp[0]), Pp[1])
    C = np.append(C, 1)
    return C


#
#
def computeEpipole(proj, C):
    # Compute the epipole
    # project C into the other camera's image plane
    e = np.matmul(proj, C)
    print("e", e)
    return e


#
#
def computeFundamentalMat(e, proj_r, proj_l):
    # Compute the fundamental matrix
    # Create the skew symmetric "cross product" matrix erx from the right epipole e (Look this up in Dirks slides)
    # Multiply with proj_r and proj_l inverse (remember to use np.linalg.pinv for pseudo inverse) to get F_lr
    ex = np.zeros((3, 3))
    ex[0, 1] = -e[2]
    ex[0, 2] = e[1]
    ex[1, 0] = e[2]
    ex[1, 2] = -e[0]
    ex[2, 0] = -e[1]
    ex[2, 1] = e[0]
    F_lr = np.matmul(np.matmul(ex, proj_r), np.linalg.pinv(proj_l))
    return F_lr


#
#
def computeEpipolarLine(F, m):
    # ------YOUR CODE GOES HERE-----------
    # Compute the epipolar line
    # ------------------------------------
    e_line = np.matmul(F, m)
    return e_line


#
#
def drawEpipolarLine(img, line, width):
    x, y, z = line[0, 0], line[1, 0], line[2, 0]
    p1 = (0, int(-z / y))
    p2 = (int(width), int(-width * x / y - z / y))

    cv2.line(img, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)


#
#
def computePluckerLine(M1, M2):
    # Compute the plucker line through two points M1 and M2
    v = 1 / np.linalg.norm(M2)
    plucker = (M2 * v, np.cross(M1, M2) * v)

    return plucker


def project2Inf(P, m):
    M_inf = np.matmul(np.linalg.inv(P), m)
    M_inf = np.append(M_inf, 0)
    return M_inf


def computePluckerIntersect(plucker_1, plucker_2):
    v1, mu1 = plucker_1
    v2, mu2 = plucker_2

    v1_cross_v2 = np.cross(v1, v2)
    norm_sq_v1_cross_v2 = np.dot(v1_cross_v2, v1_cross_v2)

    part1 = np.dot(v1, np.cross(v2, mu2))
    part2 = (np.dot(v1, v2)) * np.dot(v1, np.cross(v2, mu1))

    M1 = ((part1 - part2) / norm_sq_v1_cross_v2) * v1 + np.cross(v1, mu1)

    part1_M2 = np.dot(v2, np.cross(v1, mu1))
    part2_M2 = (np.dot(v2, v1)) * np.dot(v2, np.cross(v1, mu2))
    M2 = ((part1_M2 - part2_M2) / norm_sq_v1_cross_v2) * v2 + np.cross(v2, mu2)

    intersection = (M1 + M2) / 2
    intersection = np.append(intersection, 1)
    return intersection


#
#
def main():
    # Load images and calibration
    img_l, img_r, stereoPair = loadImagesAndCalibration()
    if img_l is None or img_r is None or stereoPair is None:
        print("Input error")
        return

    # Construct projection matrices
    proj_l = constructProjectionMat(stereoPair.cam1)
    proj_r = constructProjectionMat(stereoPair.cam2)
    print("Camera 1 (left) projection matrix", proj_l)
    print("Camera 2 (right) projection matrix", proj_r)

    # Compute optical centers
    Pp_l = splitPp(proj_l)
    Pp_r = splitPp(proj_r)
    C_l = computeOpticalCenter(Pp_l)
    C_r = computeOpticalCenter(Pp_r)
    print("Optical center Left:", C_l)
    print("Optical center Right:", C_r)
    #
    # # Compute epipoles
    e_l = computeEpipole(proj_l, C_r)
    e_r = computeEpipole(proj_r, C_l)
    print("Left epipole:", e_l)
    print("Right epipole:", e_r)
    #
    # # Compute fundamental matrix
    F_lr = computeFundamentalMat(e_r, proj_r, proj_l)
    print("Fundamental matrix left to right:", F_lr)
    #
    # # Get mouse click on a point in the left image
    m_l = getMouseClick("LeftImage", img_l)
    #
    # # Compute epipolar line in the right image, and draw it
    e_line_r = computeEpipolarLine(F_lr, m_l)
    drawEpipolarLine(img_r, e_line_r, stereoPair.cam2.image_width)
    #
    # # Get mouse click on the same point in the right image
    m_r = getMouseClick("RightImage", img_r)
    #
    # Project the points to infinity
    M_inf_l = project2Inf(Pp_l[0], m_l)
    M_inf_r = project2Inf(Pp_r[0], m_r)
    print("M_inf left:", M_inf_l)
    print("M_inf right:", M_inf_r)

    # Compute the plucker lines and their intersection
    plucker_l = computePluckerLine(C_l[:3], M_inf_l[:3])
    plucker_r = computePluckerLine(C_r[:3], M_inf_r[:3])
    print("Plucker line left:", plucker_l)
    print("Plucker line right:", plucker_r)
    intersect = computePluckerIntersect(plucker_l, plucker_r)

    print("Plucker intersection / Triangulated point:", intersect)


if __name__ == "__main__":
    main()

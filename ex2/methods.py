import cv2
import numpy as np

tmp_x = -1
tmp_y = -1


# Callback event for detecting mouse-click on images
def CallBackFunc(event, x, y, flags, userdata):
    global tmp_x, tmp_y
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp_x = x
        tmp_y = y


class Camera:
    def __init__(self):
        self.intrinsic = np.zeros((3, 3), dtype=np.float64)
        self.transformation = np.eye(4, dtype=np.float64)
        self.distortion = np.zeros((4, 1), dtype=np.float64)
        self.projection = np.zeros((4, 4), dtype=np.float64)
        self.translation = np.zeros((3, 1), dtype=np.float64)
        self.rotation = np.zeros((3, 3), dtype=np.float64)
        self.image_width = 0
        self.image_height = 0

    def printData(self):
        print(self.image_width, self.image_height)
        print(self.intrinsic)
        print(self.distortion)
        print(self.transformation)
        print(self.projection)


class StereoPair:
    def __init__(self):
        self.cam1 = Camera()
        self.cam2 = Camera()


def loadCamFromStream(input, cam):
    # Read image width and height
    cam.image_width, cam.image_height = [float(x) for x in input.readline().split()]

    # Ensure intrinsic matrix has 3x3 elements
    cam.intrinsic = np.array(
        [[float(x) for x in input.readline().split()[:3]] for _ in range(3)],
        dtype=np.float64,
    )

    # Ensure distortion has 4 elements
    cam.distortion = np.array(
        [float(x) for x in input.readline().split()[:4]], dtype=np.float64
    ).flatten()

    # Ensure rotation matrix has 3x3 elements
    cam.rotation = np.array(
        [[float(x) for x in input.readline().split()[:3]] for _ in range(3)],
        dtype=np.float64,
    )

    # Ensure translation vector has 3 elements
    cam.translation = np.array(
        [float(x) for x in input.readline().split()[:3]], dtype=np.float64
    ).reshape((3, 1))

    # Construct the transformation matrix (4x4)
    cam.transformation = np.hstack((cam.rotation, cam.translation))
    cam.transformation = np.vstack(
        (cam.transformation, np.array([0, 0, 0, 1], dtype=np.float64))
    )

    # Construct the projection matrix (3x4)
    tmp = np.hstack((cam.intrinsic, np.zeros((3, 1), dtype=np.float64)))
    cam.projection = np.dot(tmp, cam.transformation)


def readStereoCameraFile(fileNameP, stereoPair):
    with open(fileNameP, "r") as ifs:
        number_of_cameras = int(ifs.readline())
        if number_of_cameras == 2:
            loadCamFromStream(ifs, stereoPair.cam1)
            loadCamFromStream(ifs, stereoPair.cam2)
            return True
    return False


def getMouseClick(imageName, image):
    global tmp_x, tmp_y
    cv2.imshow(imageName, image)
    cv2.setMouseCallback(imageName, CallBackFunc)

    while tmp_x < 0:
        cv2.waitKey(100)

    m = np.array([[tmp_x], [tmp_y], [1]], dtype=np.float64)

    tmp_x = -1
    tmp_y = -1
    return m


# def loadImagesAndCalibration(args):
#     if len(args) != 4:
#         print("Invalid arguments")
#         print("Program usage: ./Stereopsis calibrationFile.txt leftImage.png rightImage.png")
#         return None, None, None
#
#     calibrationFile, leftImg, rightImg = args[1], args[2], args[3]
#
#     img_l = cv2.imread(leftImg)
#     img_r = cv2.imread(rightImg)
#
#     if img_l is None or img_r is None:
#         print("Error loading the images")
#         return None, None, None
#
#     stereoPair = StereoPair()
#
#     if not readStereoCameraFile(calibrationFile, stereoPair):
#         print("Error opening calibration file. Calibration file must be in old OpenCV format.")
#         return None, None, None
#
#     return img_l, img_r, stereoPair


def loadImagesAndCalibration():
    path = "./artificialImages/non-rectified"
    calibrationFile, leftImg, rightImg = (
        path + "/calibration.txt",
        path + "/left.png",
        path + "/right.png",
    )

    img_l = cv2.imread(leftImg)
    img_r = cv2.imread(rightImg)

    if img_l is None or img_r is None:
        print("Error loading the images")
        return None, None, None

    stereoPair = StereoPair()

    if not readStereoCameraFile(calibrationFile, stereoPair):
        print(
            "Error opening calibration file. Calibration file must be in old OpenCV format."
        )
        return None, None, None

    return img_l, img_r, stereoPair

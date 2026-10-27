import cv2
import numpy as np

# Global variables to store mouse click coordinates
tmp_x = -1
tmp_y = -1

# Callback function to detect mouse click on images
def CallBackFunc(event, x, y, flags, param):
    global tmp_x, tmp_y # Declare global to modify the outer variables
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp_x = x
        tmp_y = y

def getMouseClick(image_name, image):
    global tmp_x, tmp_y # Declare global to access the outer variables

    # Scale down the image so it fits the screen
    display_scale = 2  # Change this depending on how much you want to scale down the shown image
    coor_scale = 1

    # Scale down the image using pyrDown
    for i in range(display_scale):
        image = cv2.pyrDown(image)
        coor_scale *= 2

    # Display the scaled-down image
    cv2.imshow(image_name, image)
    cv2.setMouseCallback(image_name, CallBackFunc)

    # Wait for the mouse click event
    while tmp_x < 0:
        cv2.waitKey(100)

    # Return tuple of coordinates, scaled up to original
    m = (tmp_x * coor_scale, tmp_y * coor_scale)

    # Reset the coordinates after the click
    tmp_x = -1
    tmp_y = -1

    return m

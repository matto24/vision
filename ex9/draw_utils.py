import cv2
import numpy as np

def draw_cross(img, center, color, size):
    """Draws a cross centered at the specified point."""
    cv2.line(img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, 2, cv2.LINE_AA)
    cv2.line(img, (center[0] + size, center[1] - size), (center[0] - size, center[1] + size), color, 2, cv2.LINE_AA)

def draw_crosses(img, meas, gt, kf1, kf2, kf3):
    """Draws crosses for the current measurement and Kalman filter points."""
    draw_cross(img, gt, (0, 255, 0), 5)         # Green for ground truth
    draw_cross(img, meas, (0, 0, 255), 5)       # Red for measurements
    draw_cross(img, kf1, (255, 0, 0), 5)        # Blue for KF1
    draw_cross(img, kf2, (255, 165, 0), 5)      # Orange for KF2
    draw_cross(img, kf3, (75, 0, 130), 5)       # Indigo for KF3

def draw_trajec():
    """Creates a blank image for drawing trajectories."""
    return np.zeros((500, 1000, 3), dtype=np.uint8)

def draw_points(img, points, color, label=None):
    """Draws all points in the provided list onto the image."""
    for point in points:
        cv2.circle(img, point, 2, color, -1)
    
    return img

def draw_error(meas, kf1, kf2, kf3):
    """Draws error information on a blank image."""
    img = np.zeros((300, 1000, 3), dtype=np.uint8)
    draw_line(img, meas, kf1, kf2, kf3)
    return img

def draw_line(img, points, color):
    """Draw lines connecting each point in the list."""
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, 1)

def draw_lines(img, gt, meas, kf1, kf2, kf3):
    # Draw trajectories
    draw_line(img, gt, (0, 255, 0))      # Green for ground truth
    draw_line(img, meas, (0, 0, 255))    # Red for measurements
    draw_line(img, kf1, (255, 0, 0))     # Blue for KF1
    draw_line(img, kf2, (255, 165, 0))   # Orange for KF2
    draw_line(img, kf3, (75, 0, 130))    # Indigo for KF3
    return img

def draw_legend(img):
    """Adds a legend to the image."""
    cv2.putText(img, "Legend:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Ground Truth", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Measurements", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "KF1", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img, "KF2", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    cv2.putText(img, "KF3", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 0, 130), 1)
    return img

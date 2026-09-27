# ========== ARUCO TRACKING ==========

import cv2
import numpy as np

def find_aruco(image: cv2.typing.MatLike):
    # check for the image
    if image is None:
        print("not found the image")
        return None, None

    # load the aruco marker dictionary (4x4, 100 markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    # load parameters for marker detection
    parameters = cv2.aruco.DetectorParameters()
    # create aruco detector instance
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # turn image to grayscale to help detector process faster
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect markers
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids

def aruco_tracker(image: cv2.typing.MatLike, target_id: int = None):
    corners, ids = find_aruco(image)

    # if no markers are detected in the image
    if ids is None:
        return None

    # flatten ids array
    flat_ids = ids.flatten()

    for marker_id, marker_corners in zip(flat_ids, corners):
        # if target_id is specified, ignore other markers
        if target_id is not None and marker_id != target_id:
            continue

        # reshape corners into 4 points
        pts = marker_corners.reshape((4, 2))
        tl, tr, br, bl = pts

        # find center coordinate using mean
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))

        # calculate forward direction vector from bottom-left to top-left
        dx = tl[0] - bl[0]
        dy = tl[1] - bl[1]

        # calculate heading angle in radians
        theta_rad = np.arctan2(dy, dx)

        # return structured data for downstream tasks
        # robot_pose format: (cx, cy, theta_rad)
        return {
            "id": int(marker_id),
            "center": (cx, cy),
            "theta_rad": float(theta_rad),
            "theta_deg": float(np.degrees(theta_rad)),
            "corners": pts
        }

    return None

def get_scale_factor(image: cv2.typing.MatLike, physical_aruco_size: float):
    corners, ids = find_aruco(image)

    if ids is not None:
        # take the first detected marker to calculate scale
        pts = corners[0].reshape((4, 2))
        tl, tr, br, bl = pts

        # get side length in pixels
        d_top = np.linalg.norm(tr - tl)
        d_right = np.linalg.norm(br - tr)
        d_bottom = np.linalg.norm(bl - br)
        d_left = np.linalg.norm(tl - bl)
        side_in_pixel = (d_top + d_right + d_bottom + d_left) / 4.0

        # return pixels per physical unit (e.g., px/mm)
        scale_px_per_unit = side_in_pixel / physical_aruco_size
        return scale_px_per_unit

    return None

# ======================================
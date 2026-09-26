# ========== ARUCO TRACKING ==========

import cv2
import numpy as np

def find_aruco(image: cv2.typing.MatLike):
    # Check for the image
    if image is None:
        print("Not found the marker")

    # Load the arUco marker which has a size be 4 x 4 grid and marker id 0-99 (All 100 patterns)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    # Load a config for finding algorithm
    parameters = cv2.aruco.DetectorParameters()
    # Create a detector that can detect 4 x 4 grid arUco marker id 0-99 and load with default config
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Turn image to gray-scale help program find the marker easier
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find aruco marker
    # corners : keep for all corners of marker program founded
    # ids : keep for all marker id of marker program founed
    # _ (rejected) : keep for all the markers proram can't decode its bits code
    corners, ids, _ = detector.detectMarkers(gray)

    return corners, ids

def aruco_tracker(image: cv2.typing.MatLike):
    corners, ids = find_aruco(image)

    # If program founded at least one marker in image
    if ids is not None:
        # Make array ids to 1 dimension and use zip pairing each id with its corner position
        # ex ids = [[43], [12]] -> ids = [43, 12] then pair id with its corner position
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            # Four corners point ordered: top-left to top-right to bottom-right to bottom-left
            # Reshape function makes corners become a array with 4 rows and each row has two columns 
            pts = marker_corners.reshape((4, 2))
            tl, _, _, bl = pts

            # Find average of x and y position those are the center coordinate
            # pts[:, 0] means use every rows data at index 0 only (x-coordinate)
            cx = int(np.mean(pts[:, 0]))
            # pts[:, 1] means use every rows data at index 1 only (y-coordinate)
            cy = int(np.mean(pts[:, 1]))

            # Top corner vector from bottom-left to top-left
            dx = tl[0] - bl[0]
            dy = tl[1] - bl[1]


            theta_rad = np.arctan2(dy, dx)
            theta_deg = np.degrees(theta_rad)

            # calculate the end point of the line 
            line_length = 50
            end_x = int(cx + line_length * np.cos(theta_rad))
            end_y = int(cy + line_length * np.sin(theta_rad))

            # draw a red line
            cv2.line(image, (cx, cy), (end_x, end_y), (0, 0, 255), 2)

            print(f"Marker ID: {marker_id}")
            print(f"Position (X, Y): ({cx}, {cy}) pixels")
            print(f"Theta: {theta_deg:.2f}° ({theta_rad:.4f} rad)\n")

            # draw a circle on the center of marker
            cv2.aruco.drawDetectedMarkers(image, corners)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("test", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_scale_factor(image: cv2.typing.MatLike, physical_aruco_size):
    corners, ids = find_aruco(image)

    # If program founded at least one marker in image
    if ids is not None:
        # Make array ids to 1 dimension and use zip pairing each id with its corner position
        # ex ids = [[43], [12]] -> ids = [43, 12] then pair id with its corner position
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            # Four corners point ordered: top-left to top-right to bottom-right to bottom-left
            # Reshape function makes corners become a array with 4 rows and each row has two columns 
            pts = marker_corners.reshape((4, 2))
            tl, tr, br, bl = pts
    
            # get the pixel size each side from Euclidian distance and use the average to reduce the noise
            d_top = np.linalg.norm(tr - tl)
            d_right = np.linalg.norm(br - tr)
            d_bottom = np.linalg.norm(bl - br)
            d_left = np.linalg.norm(tl - bl)
            side_in_pixel = (d_top + d_right + d_bottom + d_left) / 4.0

            # return the ratio between pixel and physical size
            return side_in_pixel / physical_aruco_size

    # If there is no trackable marker in the frame
    return None

# ====================================== 
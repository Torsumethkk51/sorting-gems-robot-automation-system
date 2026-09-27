# ========== DYNAMIC OBSTACLE DETECTOR ==========

import cv2
import numpy as np

def detect_dynamic_obstacles(image: cv2.typing.MatLike, robot_corners=None, robot_radius_px: int = 25):
    h, w = image.shape[:2]
    total_image_area = h * w

    # 1.convert to grayscale to detect lines regardless of color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2.sample floor brightness from corners to avoid gems
    corner_samples = np.concatenate([
        blurred[10:int(h*0.2), 10:int(w*0.2)].flatten(),
        blurred[10:int(h*0.2), int(w*0.8):-10].flatten(),
        blurred[int(h*0.8):-10, 10:int(w*0.2)].flatten()
    ])
    floor_val = np.median(corner_samples)

    # 3.extract obstacle lines that deviate significantly from floor brightness
    diff = cv2.absdiff(blurred, int(floor_val))
    _, raw_diff_mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)

    # 4.mask out aruco marker so it is not treated as an obstacle
    if robot_corners is not None:
        pts = np.int32(robot_corners)
        cv2.fillPoly(raw_diff_mask, [pts], 0)

    # 5.filter contours: separate lines from full-field boundary and compact objects
    contours, hierarchy = cv2.findContours(raw_diff_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clean_wall_mask = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        # if contour covers almost the entire field, it is the outer boundary perimeter
        # do not fill it entirely; only draw its border edge
        if area > 0.4 * total_image_area:
            cv2.drawContours(clean_wall_mask, [cnt], -1, 255, thickness=6)
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)

        # wall lines have low circularity or high aspect ratio
        if circularity < 0.28 or aspect_ratio > 2.5:
            # draw as solid line / contour without filling massive inner spaces
            cv2.drawContours(clean_wall_mask, [cnt], -1, 255, thickness=-1)

    # 6.close tiny gaps along lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_wall_mask = cv2.morphologyEx(clean_wall_mask, cv2.MORPH_CLOSE, kernel_close)

    # 7.inflate obstacle boundaries for safe clearance
    kernel_size = int(robot_radius_px * 2)
    kernel_inflate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated_wall_mask = cv2.dilate(clean_wall_mask, kernel_inflate)

    return clean_wall_mask, inflated_wall_mask

# ===============================================
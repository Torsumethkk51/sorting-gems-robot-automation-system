# ========== OBSTACLE VISUALIZER ==========

import cv2
import numpy as np

def draw_obstacles(image: cv2.typing.MatLike, raw_wall_mask: np.ndarray, inflated_wall_mask: np.ndarray):
    # 1.create output canvas
    vis_img = image.copy()

    # 2.draw safety clearance zone (inflated area) with semi-transparent red
    # overlay creates a transparent mask layer
    overlay = vis_img.copy()
    overlay[inflated_wall_mask > 0] = (0, 0, 200)
    # blend overlay with 30% opacity so background objects stay visible
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

    # 3.draw physical wall lines with solid bright red
    vis_img[raw_wall_mask > 0] = (0, 0, 255)

    return vis_img

# =========================================
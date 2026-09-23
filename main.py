# ========== MAIN PROGRAM ==========

import cv2
from lib import vision_cleansing as vc
from lib import aruco_tracking as at

if __name__ == "__main__":
    image = cv2.imread("test-image/field.png")
    test_arUco_image = cv2.imread("test-image/field_aruco_1.png")

    if image is not None:
        cropped_image = vc.vision_cleansing(image=test_arUco_image)
        at.aruco_tracker(image=cropped_image)
        print(at.get_scale_factor(image=cropped_image, physical_aruco_size=8))
    
# ==================================
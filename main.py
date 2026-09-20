# ========== MAIN PROGRAM ==========

import cv2
from lib import vision_cleansing as vc

if __name__ == "__main__":
    image = cv2.imread("test-image/field.png")

    if image is not None:
        vc.vision_cleansing(image=image)
    
# ==================================
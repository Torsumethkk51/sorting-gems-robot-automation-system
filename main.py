# ========== MAIN PROGRAM ==========

import cv2
from lib import vision_cleansing as vc
from lib import aruco_tracking as at
from lib import color_filtering as cf
from lib import object_detector as od

if __name__ == "__main__":
    image = cv2.imread("test-image/field.png")
    test_arUco_image = cv2.imread("test-image/field_aruco_1.png")
    field_with_drop_zone = cv2.imread("test-image/field_with_drop_zone_obstacle_aruco.png")

    if image is not None:
        cropped_image = vc.vision_cleansing(image=field_with_drop_zone)
        at.aruco_tracker(image=cropped_image)
        scale_factor = at.get_scale_factor(image=cropped_image, physical_aruco_size=8)

        current_color_range = cf.color_filtering(image=cropped_image)

        annotated_img, drop_zones, gems = od.detect_objects(cropped_image, current_color_range)

        print("=== DROP ZONES ===")
        for color, pos in drop_zones.items():
            print(f"{color}: {pos}")

        print("\n=== GEMS ===")
        for color, pos_list in gems.items():
            print(f"{color} (จำนวน {len(pos_list)} ก้อน): {pos_list}")

        # แสดงผลภาพที่วาดกรอบเขียวแล้ว
        cv2.imshow("Detected Objects", annotated_img)

        # ต้องมีคำสั่งนี้เพื่อให้หน้าต่างค้างไว้จนกว่าจะกดปุ่มใดๆ (หรือใส่ 1 ถ้าอยู่ใน while loop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
# ==================================
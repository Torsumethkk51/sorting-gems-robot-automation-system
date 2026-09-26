# ========== OBJECT DETECTOR ==========

import cv2
import numpy as np

def detect_objects(
    image: cv2.typing.MatLike,
    color_ranges: dict,
    dropzone_min_area: float = 6000.0,
    gem_min_area: float = 120.0,
    gem_max_area: float = 4500.0
):
    """
    แยกวัตถุออกเป็น drop_zones และ gems โดยแสดงผลเป็นกล่องสี่เหลี่ยมสีเขียวทั้งคู่
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    annotated_image = image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    drop_zones = {}
    gems = {}

    for color_name, ranges in color_ranges.items():
        gems[color_name] = []
        drop_zones[color_name] = None
        
        # 1. รวม Mask ทุกช่วง
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            sub_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            combined_mask = combined_mask | sub_mask

        # 2. ลด Noise
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        # 3. หา Contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            # --- หมวด Drop Zone (วงใหญ่) ---
            if area >= dropzone_min_area:
                drop_zones[color_name] = (cx, cy)
                
                # กล่องเขียว + จุดแดงตรงกลาง + ป้ายกำกับ
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(annotated_image, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(annotated_image, f"{color_name} ({cx},{cy})", 
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # --- หมวด Gems (หินก้อนเล็ก) ---
            elif gem_min_area <= area <= gem_max_area:
                gems[color_name].append((cx, cy))
                
                # กล่องเขียว + จุดแดงตรงกลาง + ป้ายกำกับ
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(annotated_image, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(annotated_image, f"{color_name} [{cx},{cy}]", 
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return annotated_image, drop_zones, gems

# =====================================
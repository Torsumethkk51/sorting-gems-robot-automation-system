# ========== VISION CLEANSING ==========

import cv2

def vision_cleansing(image: cv2.typing.MatLike):
    # 1.Turn vision into gray-scale    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2.Apply gaussian-blurred into gray image
    # Blur by averaging 5px around
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3.Apply Canny's algorithm to detect all objects edge
    # 50 is difference of color which we ignore
    # 150 is difference of color which is a edge
    edges = cv2.Canny(blurred_image, 50, 150)

    # 4.Find the largest shape area to find the field
    # 4.1.Find contours from edges
    # RETR_EXTERNAL is find contours mode to find just the extreme outer boundary
    # CHAIN_APPROX_SIMPLE is keep only point to point not every point, it's useful for memory saving
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4.2.Find the largest contour by decending ordered sorting
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # After sorting the largest contour will stay at index 0
    largest_contour = sorted_contours[0]

    # 5.Crop the image

    # Use minAreaRect -> Just find the fittest coverable rectangle and allow to rotate the boundary
    rect = cv2.minAreaRect(largest_contour)
    # Deconstruct rect in to these values -> (cx, cy) = center of the box, (w, h) = size of the box, angle = the rotation angle relative to x-axis
    (cx, cy), (w, h), angle = rect

    # CV2 don't which side is width or height
    # Force w must always greater than h
    if w < h:
        # If w < h : Swap w and h values then add angle by 90 because in CV2 will return angle in interval of (-90, 0] only
        # From formula dtheta = 180 - 90 - theta => dtheta = 90 - theta
        # But theta <= 0 means dtheta = 90 + theta
        angle += 90
        r = w
        w = h
        h = r

    # Calculate the rotation matrix whereby, the rotation center must be the center of the box
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Find image size by sliced image array [height, width, channels] just first two values
    (img_h, img_w) = image.shape[:2]

    # Rotate the image and keep the image original size
    rotated_image = cv2.warpAffine(image, matrix, (img_w, img_h))

    # Find top left x of the box to wrapped image in x-axis
    # btlx means box top left x
    btlx = cx - w / 2

    # Find top right y of the box to wrapped image in y-axis
    # btly means box top right y
    btry = cy - h / 2

    # Rounded them because pixel amount must be an integer
    btlx = int((round(btlx)))
    btry = int((round(btry)))

    # X and Y use to add with btlx and btry, then them must be a integer too
    w = int(round(w))
    h = int(round(h))

    # Wrap from 
    # Y-axis -> box top right y to box top right y + h
    # X-axis -> box top left x to box top left x + w
    # Do y first because computer use row (y) come first
    cropped_image = rotated_image[btry : btry + h, btlx : btlx + w]

    # Show result preview after that return the cropped image
    cv2.imshow("Result preview", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_image;

# ====================================== 
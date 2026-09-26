# ========== COLOR FILTERING ==========

from pathlib import Path
import cv2
import numpy as np
import json
import os

# specifier to project path
project_dir = Path(__file__).resolve().parent.parent

# define the hsv values config file location in project folder
config_file = project_dir / "config" / "hsv_config.json"

# define all colors on the field
colors = ["blue", "cyan", "green", "orange", "purple", "red"]

# when json file is not ready, program will use default config
default_color_range = {
    'blue': [
        [[77, 151, 75], [103, 255, 200]]
    ],
    'cyan': [
        [[85, 83, 176], [100, 255, 255]]
    ],
    'green': [
        [[40, 35, 129], [78, 255, 205]]
    ],
    'orange': [
        [[11, 110, 142], [22, 255, 255]]
    ],
    'purple': [
        [[130, 40, 40], [160, 255, 255]]
    ],
    'red': [
        [[0, 90, 125], [5, 255, 255]],
        [[178, 113, 126], [180, 255, 255]]
    ]
}

# an empty function add because it is the rule of cv2 createTracker we have to send the callback function which run when tracker has change but we have tracker update already
def nothing(x):
    pass

def update_trackbars(color_idx, layer_idx, data):
    # get color name which want to update
    color_name = colors[color_idx]
    # get the hsv ranges from the specific color name
    ranges = data[color_name]
    
    # the layer trackbar will back to zero when that color hasn't enough layer
    actual_layer = 0 if layer_idx >= len(ranges) else layer_idx

    # get the range of that layer of that color
    lower = ranges[actual_layer][0]
    upper = ranges[actual_layer][1]

    # update trackbars
    cv2.setTrackbarPos('H Min', 'HSV Tuner', lower[0])
    cv2.setTrackbarPos('S Min', 'HSV Tuner', lower[1])
    cv2.setTrackbarPos('V Min', 'HSV Tuner', lower[2])
    cv2.setTrackbarPos('H Max', 'HSV Tuner', upper[0])
    cv2.setTrackbarPos('S Max', 'HSV Tuner', upper[1])
    cv2.setTrackbarPos('V Max', 'HSV Tuner', upper[2])

def color_filtering(image: cv2.typing.MatLike):
    # current color range must use default first
    current_color_range = default_color_range.copy()

    # if json is exist and json size is more than 0 just load them then set to current color range but if it failed just use default value
    if os.path.exists(config_file) and os.path.getsize(config_file) > 0:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                if loaded_data:  # check for not empty json
                    current_color_range = loaded_data
        except (json.JSONDecodeError, OSError):
            current_color_range = default_color_range.copy()

    # load hsv tuner window
    cv2.namedWindow('HSV Tuner')

    # set its size
    cv2.resizeWindow('HSV Tuner', 450, 400)

    # add each trackbar into the window
    cv2.createTrackbar('Color (0-5)', 'HSV Tuner', 0, len(colors) - 1, nothing)
    cv2.createTrackbar('Layer (0-1)', 'HSV Tuner', 0, 1, nothing)
    cv2.createTrackbar('H Min', 'HSV Tuner', 0, 179, nothing)
    cv2.createTrackbar('S Min', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('V Min', 'HSV Tuner', 0, 255, nothing)
    cv2.createTrackbar('H Max', 'HSV Tuner', 179, 179, nothing)
    cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)
    cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)

    # initialize trackbar at first color and first layer
    current_idx = 0
    current_layer = 0
    update_trackbars(current_idx, current_layer, current_color_range)

    # load image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # while in tuning don't close the window
    while True:
        # check chaning of color or layer of tracker
        selected_idx = cv2.getTrackbarPos('Color (0-5)', 'HSV Tuner')
        selected_layer = cv2.getTrackbarPos('Layer (0-1)', 'HSV Tuner')

        # check does current color has current layer if not just force it value to layer 0
        total_layers = len(current_color_range[colors[selected_idx]])
        if selected_layer >= total_layers:
            selected_layer = 0
            cv2.setTrackbarPos('Layer (0-1)', 'HSV Tuner', 0)

        # if color or layer has change then update trackbars
        if selected_idx != current_idx or selected_layer != current_layer:
            current_idx = selected_idx
            current_layer = selected_layer
            update_trackbars(current_idx, current_layer, current_color_range)

        # read values from trackbars
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')

        # update values on the current color range at specific color and layer
        current_color = colors[current_idx]
        current_color_range[current_color][current_layer] = [[h_min, s_min, v_min], [h_max, s_max, v_max]]

        # combine all mask
        # start all mask with black bg
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        # each hsv fileter from current color mask to combined mask
        for lower, upper in current_color_range[current_color]:
            sub_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            combined_mask = combined_mask | sub_mask

        # mask image 
        result = cv2.bitwise_and(image, image, mask=combined_mask)
        # write color and layer status into the image
        cv2.putText(result, f"Color: {current_color} (Layer {current_layer+1}/{total_layers})", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('HSV Tuner Result', result)

        # wait for key q (quit), s (save into json file)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(current_color_range, f, indent=4)
            print(f"Save into {config_file} completely")

    cv2.destroyAllWindows()

# ====================================
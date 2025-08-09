import cv2
import sys
import numpy as np

PREVIEW  = 0 # Preview mode
BLUR     = 1 # Blurring mode
FEATURES = 2 # Corner feature detector
CANNY    = 3 # Canny edge detector

feature_params = dict(maxCorners = 500, qualityLevel = 0.2, minDistance = 15, blockSize = 9)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive: bool = True # parameter controlling program loop

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)
    
    # display unfiltered
    if image_filter == PREVIEW:
        result = frame
    # use canny edge detection
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 80, 150)
    # blur the image
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    # display features with circles
    elif image_filter == FEATURES:
        result = frame
        
        # create the gray frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # select features based on the thresholds set above
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)

        # draw every corner circle
        if corners is not None:
            # ints are necessary for openCV points
            for x, y in np.float32(corners).reshape(-1, 2):
                cv2.circle(result, center = (int(x), int(y)), radius = 10, color = (0, 255, 0), thickness = 1)

    # display the result
    cv2.imshow(win_name, result)

    # handle keys for user input
    key = cv2.waitKey(1)
    
    # quit
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    # canny
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    # blur
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    # features
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    # preview
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

# release resources
source.release()
cv2.destroyWindow(win_name)

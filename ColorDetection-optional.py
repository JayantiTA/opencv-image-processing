import cv2 as cv
import numpy as np

# optional argument for trackbar parameter
def nothing():
    pass
 
# capture video from the stream
webcam = cv.VideoCapture(0)
cv.namedWindow('Colorbars Background')
cv.namedWindow('Colorbars Object')
 
# assign strings for ease of coding
hue_high    = 'Hue High'
hue_low     = 'Hue Low'
sat_high    = 'Saturation High'
sat_low     = 'Saturation Low'
val_high    = 'Value High'
val_low     = 'Value Low'
area_min    = 'Area Minimum'
window      = 'Colorbars Background'
window2     = 'Colorbars Object'

# begin creating trackbars for background
cv.createTrackbar(hue_low, window, 0, 179, nothing)
cv.createTrackbar(hue_high, window, 0, 179, nothing)
cv.createTrackbar(sat_low, window, 0, 255, nothing)
cv.createTrackbar(sat_high, window, 0, 255, nothing)
cv.createTrackbar(val_low, window, 0, 255, nothing)
cv.createTrackbar(val_high, window, 0, 255, nothing)
cv.createTrackbar(area_min, window, 0, 5000, nothing)

# begin creating trackbars for object
cv.createTrackbar(hue_low, window2, 0, 179, nothing)
cv.createTrackbar(hue_high, window2, 0, 179, nothing)
cv.createTrackbar(sat_low, window2, 0, 255, nothing)
cv.createTrackbar(sat_high, window2, 0, 255, nothing)
cv.createTrackbar(val_low, window2, 0, 255, nothing)
cv.createTrackbar(val_high, window2, 0, 255, nothing)
cv.createTrackbar(area_min, window2, 0, 5000, nothing)

# start a while loop
while True:
    # read the video capture
    _, image_frame = webcam.read()

    # convert color to HSV 
    hsv_frame = cv.cvtColor(image_frame, cv.COLOR_BGR2HSV)

    # get trackbar positions for background
    hue_low_background  = cv.getTrackbarPos(hue_low, window)
    hue_high_background = cv.getTrackbarPos(hue_high, window)
    sat_low_background  = cv.getTrackbarPos(sat_low, window)
    sat_high_background = cv.getTrackbarPos(sat_high, window)
    val_low_background  = cv.getTrackbarPos(val_low, window)
    val_high_background = cv.getTrackbarPos(val_high, window)
    area_min_background = cv.getTrackbarPos(area_min, window)

    # get trackbar positions for object
    hue_low_object  = cv.getTrackbarPos(hue_low, window2)
    hue_high_object = cv.getTrackbarPos(hue_high, window2)
    sat_low_object  = cv.getTrackbarPos(sat_low, window2)
    sat_high_object = cv.getTrackbarPos(sat_high, window2)
    val_low_object  = cv.getTrackbarPos(val_low, window2)
    val_high_object = cv.getTrackbarPos(val_high, window2)
    area_min_object = cv.getTrackbarPos(area_min, window2)

    # make array from final values
    hsv_lower_background = np.array([hue_low_background, sat_low_background,
                                     val_low_background], np.uint8)
    hsv_upper_background = np.array([hue_high_background, sat_high_background,
                                     val_high_background], np.uint8)    

    hsv_lower_object = np.array([hue_low_object, sat_low_object, val_low_object],
                                 np.uint8)
    hsv_upper_object = np.array([hue_high_object, sat_high_object, val_high_object],
                                 np.uint8)    

    # define area minimum and create mask
    area_min_background = area_min_background
    background_mask     = cv.inRange(hsv_frame, hsv_lower_background, hsv_upper_background)

    area_min_object = area_min_object
    object_mask     = cv.inRange(hsv_frame, hsv_lower_object, hsv_upper_object)

    kernel = np.ones((5, 5), np.uint8)

    # result for background
    background_mask = cv.morphologyEx(background_mask, cv.MORPH_OPEN, kernel)
    res_background  = cv.bitwise_and(image_frame, image_frame,
                                     mask = background_mask)
    cv.imshow("Background Mask", background_mask)

    # result for object
    object_mask     = cv.morphologyEx(object_mask, cv.MORPH_CLOSE, kernel)
    res_object      = cv.bitwise_and(image_frame, image_frame,
                                     mask = object_mask)
    cv.imshow("Object Mask", object_mask)

    # create contour to track background
    background_contour, hierarchy = cv.findContours(background_mask,
                                                    cv.RETR_TREE,
                                                    cv.CHAIN_APPROX_SIMPLE)

    for b_contour in background_contour:
        background_area = cv.contourArea(b_contour)
        if background_area > area_min_background:
            x, y, w, h = cv.boundingRect(b_contour)
            image_frame = cv.rectangle(image_frame, (x,y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)
            
            cv.putText(image_frame, "Background", (x, y),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0,
                       (0, 0, 255))
            
            # create contour to track object
            object_contour, hierarchy = cv.findContours(object_mask,
                                                        cv.RETR_TREE,
                                                        cv.CHAIN_APPROX_SIMPLE)
        
            for o_contour in object_contour:
                object_area = cv.contourArea(o_contour)
                if object_area > area_min_object:
                    x2, y2, w2, h2 = cv.boundingRect(o_contour)
                    # if object inside background area
                    if x2 >= x and y2 >= y and x2 + w2 <= x + w and y2 + h2 <= y + h:
                        image_frame_2 = cv.rectangle(image_frame, (x2, y2),
                                                     (x2 + w2, y2 + h2),
                                                     (0, 255, 0), 2)
                        
                        cv.putText(image_frame_2, "Object", (x2, y2),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0,
                                   (0, 255, 0))

    # show result of color detection
    cv.imshow("Color detection", image_frame)

    # end loop when press 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break


    # green_lower = np.array([25, 52, 72], np.uint8)
    # green_upper = np.array([102, 255, 255], np.uint8)
    # green_mask = cv.inRange(hsv_frame, green_lower, green_upper)

    # blue_lower = np.array([94, 80, 2], np.uint8)
    # blue_upper = np.array([120, 255, 255], np.uint8)
    # blue_mask = cv.inRange(hsv_frame, blue_lower, blue_upper)

    # red_lower = np.array([136, 87, 111], np.uint8)
    # red_upper = np.array([180, 255, 255], np.uint8)
    # red_mask = cv.inRange(hsv_frame, red_lower, red_upper)
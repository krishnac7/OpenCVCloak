import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
background = 0

uh,us,uv =191,224,230      #default set to identify red,alter the hsv values to form proper filter
lh,ls,lv = 166,98,43

lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

def getBG():
    global background
    ret, background = cap.read()
    background = np.flip(background, axis=1)

def getHSV():
    global lower_hsv,upper_hsv,uh,us,uv,lh,ls,lv
    ret, image = cap.read()
    image = np.flip(image, axis=1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    window_name = "HSV Calibrator"
    cv2.namedWindow(window_name)
    def nothing(x):
        pass

    # create trackbars for Upper HSV
    cv2.createTrackbar('UpperH', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('UpperH', window_name, uh)

    cv2.createTrackbar('UpperS', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('UpperS', window_name, us)

    cv2.createTrackbar('UpperV', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('UpperV', window_name, uv)

    # create trackbars for Lower HSV
    cv2.createTrackbar('LowerH', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('LowerH', window_name, lh)

    cv2.createTrackbar('LowerS', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('LowerS', window_name, ls)

    cv2.createTrackbar('LowerV', window_name, 0, 255, nothing)
    cv2.setTrackbarPos('LowerV', window_name, lv)

    font = cv2.FONT_HERSHEY_SIMPLEX
    while (1):
        mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=15)
        mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=8)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=22)
        im = cv2.bitwise_and(image,image,mask=mask1)
        cv2.putText(im, 'Lower HSV: [' + str(lh) + ',' + str(ls) + ',' + str(lv) + ']', (10, 30), font, 0.5,
                   (200, 255, 155), 1, cv2.LINE_AA)
        cv2.putText(im, 'Upper HSV: [' + str(uh) + ',' + str(us) + ',' + str(uv) + ']', (10, 60), font, 0.5,
                   (200, 255, 155), 1, cv2.LINE_AA)

        cv2.imshow(window_name, im)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of Upper HSV trackbars
        uh = cv2.getTrackbarPos('UpperH', window_name)
        us = cv2.getTrackbarPos('UpperS', window_name)
        uv = cv2.getTrackbarPos('UpperV', window_name)

        # get current positions of Lower HSCV trackbars
        lh = cv2.getTrackbarPos('LowerH', window_name)
        ls = cv2.getTrackbarPos('LowerS', window_name)
        lv = cv2.getTrackbarPos('LowerV', window_name)
        upper_hsv = np.array([uh, us, uv])
        lower_hsv = np.array([lh, ls, lv])

        time.sleep(.1)

    cv2.destroyAllWindows()

time.sleep(0.3)
getBG()
while (cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=15) #Some cleaning to ensure smoothness in output
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=22)
    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background,background,mask=mask1)
    res2 = cv2.bitwise_and(img,img,mask=mask2)
    final_output = cv2.addWeighted(res1,1,res2,1,0)
    k = cv2.waitKey(1)
    if k == 27:
        break
    if k == ord('h'):
        getHSV()
    if k == ord('b'):
        getBG()
        cv2.putText(final_output, "Background Set", (300, 30), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255), 1, cv2.LINE_AA)
    cv2.namedWindow("Portal", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Portal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.putText(final_output, "1.Press B to set background", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(final_output, "2.Press H to Configure HSV", (6,45), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Portal', final_output)
import numpy as np
import cv2
import sys
from random import randint

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
#print(BORDER_COLOR)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/people.mp4'

BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
#print(BGS_TYPES[1])

def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3,3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3,3), np.uint8)
    return kernel

#get_kernel('closing')

def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations = 2)
    if filter == 'dilation':
        return cv2.dilate(img, get_kernel('dilation'), iterations = 2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations = 2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations = 2)
        dilation = cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation

def get_bgsubtractor(BGS_TYPE):
    # https://docs.opencv.org/3.4/d1/d5c/classcv_1_1bgsegm_1_1BackgroundSubtractorGMG.html
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    # https://docs.opencv.org/3.4/d6/da7/classcv_1_1bgsegm_1_1BackgroundSubtractorMOG.html
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    # https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    # https://docs.opencv.org/3.4/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    # https://docs.opencv.org/3.4/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Invalid detector!')
    sys.exit(0)

cap = cv2.VideoCapture(VIDEO_SOURCE)
# 0 = GMG, 1 = MOG2, 2 = MOG, 3 = KNN,  4 = CNT
bg_subtractor = get_bgsubtractor(BGS_TYPES[1])
BGS_TYPE = BGS_TYPES[4]
minArea = 250
maxArea = 3000

def main():
    while cap.isOpened():
        ok, frame = cap.read()
        print(ok)
        #print(frame.shape)
        if not ok:
            print('End of video or video not opened successfully.')
            break

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        bg_mask= bg_subtractor.apply(frame)
        bg_mask = get_filter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)  # kernel applying for 5*5 matrix
        (contours, hierarchy) = cv2.findContours( bg_mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        for cnt in contours:
            area =cv2.contourArea(cnt)
            if area >= minArea:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.drawContours(frame, cnt, 1, TRACKER_COLOR, 10)
                cv2.drawContours(frame,cnt,1,(255,255,255),1)

            if area >= maxArea:
                cv2.rectangle(frame, (x,y), (x+120, y-13), (49,49,49), -1)
                cv2.putText(frame, 'warning',(x,y-2), FONT,0.4,(255,255,255),1,cv2.LINE_AA)
                cv2.drawContours(frame,[cnt],-1, (0, 0, 255),2)
                cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 1)
        result = cv2.bitwise_and(frame, frame, mask=bg_mask)

        # https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html
        # https://www.pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/

        if not ok:
            print('End processing the video')
            break


        cv2.imshow('Frame', frame)
        #cv2.imshow('Mask', result)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()



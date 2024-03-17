import cv2
import sys
from random import randint

TEXTCOLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDERCOLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = "videos/people.mp4"
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[1]
# GMG 38
# MOG 28
# MOG2 18
# KNN 16
# CNT 15

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Unknown createBackgroundSubtractor type")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = getBGSubtractor(BGS_TYPE)
e1 = cv2.getTickCount() #the number of clock ticks at that particular moment

def main():
    frame_number = -1
    while (cap.isOpened):
        ok, frame = cap.read()

        if not ok:
          print('Finish processing the video')
          break

        frame_number += 1

        bg_mask = bg_subtractor.apply(frame)
        res = cv2.bitwise_and(frame, frame, mask=bg_mask)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', res)

        if cv2.waitKey(1) & 0xFF == ord("q") or frame_number > 250:
            break

    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency() #e1 is the start time, e2 is the end time, and t is the elapsed time in seconds.
    print(t) # cv2.getTickFrequency() to measure the execution time of a particular operation.

main()

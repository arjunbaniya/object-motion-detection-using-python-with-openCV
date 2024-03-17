import numpy as np
import cv2
import sys
from random import randint
import csv

fp = open('report.csv', mode='w')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

TEXT_COLOR = (randint(0, 255), randint(0,255), randint(0,255))
BORDER_COLOR = (randint(0, 255), randint(0,255), randint(0,255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 1.2
VIDEO_SOURCE = "videos/people.mp4"
TITLE_TEXT_POSITION = (100, 40)

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]

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
    print("Invalid detector")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)

bg_subtractor = []
#used to iterate over elements of BGS_TYPES and provides both the index (i) and the value (a) in each iteration.
for i, a in enumerate(BGS_TYPES):
    #print(i, a)
    bg_subtractor.append(getBGSubtractor(a))

#print(bg_subtractor)


def main():
    framecount = 0
    while cap.isOpened():
        ok, frame = cap.read()
        #print(ok)

        if not ok:
            print('Finished processing the video')
            break

        framecount += 1
        frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)

        gmg = bg_subtractor[0].apply(frame)
        mog = bg_subtractor[1].apply(frame)
        mog2 = bg_subtractor[2].apply(frame)
        knn = bg_subtractor[3].apply(frame)
        cnt = bg_subtractor[4].apply(frame)

        # count the number of non-zero elements in an array
        gmg_count = np.count_nonzero(gmg)
        mog_count = np.count_nonzero(mog)
        mog2_count = np.count_nonzero(mog2)
        knn_count = np.count_nonzero(knn)
        cnt_count = np.count_nonzero(cnt)

        #write on count pixel on csv file
        writer.writerow({'Frame': 'MOG', 'Pixel Count': mog_count})
        writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog2_count})
        writer.writerow({'Frame': 'GMG', 'Pixel Count': gmg_count})
        writer.writerow({'Frame': 'KNN', 'Pixel Count': knn_count})
        writer.writerow({'Frame': 'CNT', 'Pixel Count': cnt_count})

        cv2.putText(mog, 'MOG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(mog2, 'MOG2', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(gmg, 'GMG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(knn, 'KNN', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(cnt, 'CNT', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('Original', frame)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        cv2.imshow('KNN', knn)
        cv2.imshow('CNT', cnt)

        cv2.moveWindow('Original', 0, 0)
        cv2.moveWindow('MOG', 0, 250)
        cv2.moveWindow('KNN', 0, 500)
        cv2.moveWindow('GMG', 719, 0)
        cv2.moveWindow('MOG2', 719, 250)
        cv2.moveWindow('CNT', 719, 500)

        k = cv2.waitKey(0) & 0xff
        if k == 27:  # ESC
            break


main()


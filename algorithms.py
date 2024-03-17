import numpy as np #for numerical operations in Python
import cv2
import sys
from random import randint
import tkinter as tk
from tkinter import ttk

TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/Cars.mp4'
#bg_subtractor = cv2.createBackgroundSubtractorMOG2()

#kernel structuring is  morphological operations, specifically for dilation, opening, and closing.
def get_kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)
        return kernel
#various morphological operations to an input image (img) based on the specified filter.
def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, get_kernel('dilation'), iterations=2)
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation
# difine background subtraction techniques
def get_bgsubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Invalid detector!')
    sys.exit(0)
#creating a VideoCapture object using OpenCV to capture video from a specified source
cap = cv2.VideoCapture(VIDEO_SOURCE)

# select algorithm
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
def apply_algorithm():
    selected_algorithm = algorithm_var.get()
    global bg_subtractor
    bg_subtractor = get_bgsubtractor(selected_algorithm)

#processing all frame using while or load each frames
def process_video():
    while cap.isOpened():
        ok, frame = cap.read()
        print(frame)

        if not ok or frame is None:
            print('Error: Unable to read a valid frame from the video source.')
            break
        #resize frame
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #apply background subtraction in each frame (binnary representation)
        bg_mask = bg_subtractor.apply(frame)
        #apply morohological operation
        fg_mask_combine = get_filter(bg_mask, 'combine')

        # function performs a bitwise AND operation between two input arrays (images), element-wise.
        res_combine = cv2.bitwise_and(frame, frame, mask=fg_mask_combine)
        #font showing in frame
        cv2.putText(res_combine, 'Background subtractor: ' + algorithm_var.get(), (10, 50), FONT, 0.5, BORDER_COLOR, 3,
                    cv2.LINE_AA)
        #showing result
        cv2.imshow('Frame', frame)
        cv2.imshow('BG mask', bg_mask)

        cv2.imshow('Combine final', res_combine)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
#it is for  Gui interface for user
root = tk.Tk()
root.title("Background Subtraction Algorithm Selector")

algorithm_var = tk.StringVar()
algorithm_label = tk.Label(root, text="Select Background Subtraction Algorithm:")
algorithm_label.pack()
algorithm_dropdown = ttk.Combobox(root, textvariable=algorithm_var)
algorithm_dropdown['values'] = ('GMG', 'MOG2', 'MOG', 'KNN', 'CNT')
algorithm_dropdown.pack()

#start_button = tk.Button(root, text="Start detection", command=apply_algorithm)
#start_button.pack()

process_button = tk.Button(root, text="Start detection", command=process_video)
process_button.pack()

root.mainloop()

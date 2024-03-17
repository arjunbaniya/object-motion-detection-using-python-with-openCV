import cv2
import sys
from random import randint
import matplotlib.pyplot as plt

TEXTCOLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDERCOLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = "videos/cars.mp4"
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]

# Create a list to store performance results for each background subtractor
performance_results = []

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

def main():
    frame_number = -1
    for BGS_TYPE in BGS_TYPES:
        bg_subtractor = getBGSubtractor(BGS_TYPE)
        e1 = cv2.getTickCount() #time to load a  first frome at t1

        while (cap.isOpened):
            ok, frame = cap.read()

            if not ok:
                print(f'Finish processing the video with {BGS_TYPE}')
                break

            frame_number += 1

            bg_mask = bg_subtractor.apply(frame)
            #res = cv2.bitwise_and(frame, frame, mask=bg_mask)

            # cv2.imshow('Frame', frame)
            # cv2.imshow('Mask', res)

            if cv2.waitKey(1) & 0xFF == ord("q") or frame_number > 250:
                break

        e2 = cv2.getTickCount()
        #t will contain the time taken by the code or processing between the e1 and e2 measurements
        t = (e2 - e1) / cv2.getTickFrequency() #cv2.getTickFrequency():returns the number of clock ticks per second.
        performance_results.append((BGS_TYPE, t))

    # Print the performance results
    for result in performance_results:
        print(f"Background Subtractor {result[0]} took {result[1]} seconds")

    # Create a bar chart to visualize the performance results
    algorithms, timings = zip(*performance_results)
    fig, ax = plt.subplots()
    bars = ax.barh(algorithms, timings)
    plt.xlabel('Time (seconds)')
    plt.title('Performance Comparison of Background Subtraction Algorithms')
    plt.gca().invert_yaxis()  # Invert the y-axis to display the fastest algorithm at the top

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center')

    plt.show()

main()

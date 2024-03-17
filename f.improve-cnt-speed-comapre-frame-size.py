import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# Set up video capture
cap_original = cv2.VideoCapture('videos/people.mp4')

# Create background subtractor
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT(
    minPixelStability=15, useHistory=True, maxPixelStability=15*60, isParallel=True
)

# Lists to store processing times
processing_times_original = []
processing_times_resized = []

frame_number = 0

while cap_original.isOpened():
    ret_original, frame_original = cap_original.read()
    print(frame_original)

    if not ret_original:
        break

    frame_number += 1
   
    #orginal frame
    #     frame_resized = cv2.resize(frame_original, (0, 0), fx=1, fy=1)
    # Resize the frame to half its original size
    frame_resized = cv2.resize(frame_original, (0, 0), fx=0.5, fy=0.5)

    # Measure processing time for the original size frame
    start_time_original = time.time()
    fg_mask_original = bg_subtractor.apply(frame_original)
    end_time_original = time.time()
    processing_time_original = end_time_original - start_time_original
    processing_times_original.append(processing_time_original)

    # Measure processing time for the resized frame
    start_time_resized = time.time()
    fg_mask_resized = bg_subtractor.apply(frame_resized)
    end_time_resized = time.time()
    processing_time_resized = end_time_resized - start_time_resized
    processing_times_resized.append(processing_time_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap_original.release()
cv2.destroyAllWindows()

# Plot processing times
frame_numbers = range(1, frame_number + 1)
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, processing_times_original, label='Original Size')
plt.plot(frame_numbers, processing_times_resized, label='Resized Size')
plt.xlabel('Frame Number')
plt.ylabel('Processing Time (seconds)')
plt.title('Processing Time Comparison')
plt.legend()
plt.grid(True)
plt.show()

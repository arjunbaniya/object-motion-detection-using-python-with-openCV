import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set up video capture
VIDEO_SOURCE = "videos/test.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Create background subtractor (CNT)
bg_subtractor = cv2.bgsegm.createBackgroundSubtractorCNT()

# Lists to store pixel counts with different kernels
ellipse_kernel_counts = []
rect_kernel_counts = []

frame_count = 0
# analyzing each frames
while cap.isOpened():
    ok, frame = cap.read()
    print(frame)
    if not ok:
        print('Finished processing the video')
        break

    # Resize frame
    frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)

    # Apply background subtraction with different kernels
    cnt_ellipse = bg_subtractor.apply(frame)
    cnt_rect = bg_subtractor.apply(frame)

    # Modify the structuring element with different kernels
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply morphological closing
    cnt_ellipse = cv2.morphologyEx(cnt_ellipse, cv2.MORPH_CLOSE, kernel_ellipse)
    cnt_rect = cv2.morphologyEx(cnt_rect, cv2.MORPH_CLOSE, kernel_rect)

    # Count white pixels
    cnt_ellipse_count = np.count_nonzero(cnt_ellipse)
    cnt_rect_count = np.count_nonzero(cnt_rect)

    # Append counts to lists
    ellipse_kernel_counts.append(cnt_ellipse_count)
    rect_kernel_counts.append(cnt_rect_count)

    frame_count += 1

# Plot pixel counts using different kernels on a graph
frame_numbers = range(frame_count)
plt.figure(figsize=(10, 6))
plt.plot(frame_numbers, ellipse_kernel_counts, label='Elliptical Kernel')
plt.plot(frame_numbers, rect_kernel_counts, label='Rectangular Kernel')
plt.xlabel('Frame Number')
plt.ylabel('White Pixel Count')
plt.title('White Pixel Count Comparison with Different Kernels (CNT Algorithm)')
plt.legend()
plt.grid(True)
plt.show()

# Display the total counts
print(f'Total Count with Elliptical Kernel: {sum(ellipse_kernel_counts)} pixels')
print(f'Total Count with Rectangular Kernel: {sum(rect_kernel_counts)} pixels')

# Release video capture
cap.release()
cv2.destroyAllWindows()

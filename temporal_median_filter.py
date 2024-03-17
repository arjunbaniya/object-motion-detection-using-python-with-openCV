import numpy as np # scientific library, nampy used in some mathmetical calculation like matrixes ,vectors and some mathematical calculation
import cv2 # opencv ( open computer vision liberary, it has some feature of showing image )
print(cv2.__version__)

VIDEO_SOURCE='videos/Cars.mp4'
VIDEO_OUT = 'videos/results/temporal_median_filter.avi'

cap= cv2.VideoCapture(VIDEO_SOURCE) #videoCapture is class
# video has lot of sequence of the images

has_frame, frame= cap.read() #this funtion read first frame of video
print(has_frame, frame.shape) # True (720, 1280, 3), if you got error we need to check video path otherwise result is true and 720 and 1280 shape or dimenssion of video and 3 means number of channel(red, blue and green)

fourcc = cv2.VideoWriter_fourcc(*'XVID') #video save xvid format for avi whcih save on results

writer =cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1],frame.shape[0]), False) # higher the value faster the speed 25 use as speed, false means grayscale frame shows

#print(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #total count frame in video
#print(np.random.uniform(size = 25)) random uniform number generates
Frames_ids=cap.get(cv2.CAP_PROP_FRAME_COUNT)* np.random.uniform(size = 25)
print(Frames_ids) # this is the selected frames that used to generate a static background image

#cap.set(cv2.CAP_PROP_POS_FRAMES,388 )# capture frame on frames number 388
# has_frame, frame= cap.read();
# cv2.imshow('test', frame)
# cv2.waitKey(0)

# all frame in array 
frames= []
for fid in Frames_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    has_frame, frame= cap.read();
    frames.append(frame)
#print(np.asarray(frames).shape) #convert frames numpy array

#print(frames[1])

#show random 5 frames for static test
#for frame in frames:
#   cv2.imshow('test', frame)
#  cv2.waitKey(0)

#print(np.mean([1,3,5,6,8,9]))
#print((1+3+5+6+8+9)/6)
#print(np.median([1,3,5,6,8,9]))
#print((5+6)/2)

median_frame = np.median(frames, axis = 0).astype(dtype=np.uint8)# axis =0 means we are extracting each one of column , astype(dtype=np.uint8) means converting float to integer
#print(frame[0])

#print(median_frame)
# find median frame which used to compare background image and foreground image
#cv2.imshow('median_frame', median_frame)
#cv2.waitKey(0)
# save median image
cv2.imwrite('model_median_frame.jpg', median_frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#convert in grayscale

gray_median_frame =cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray', gray_median_frame)
#cv2.waitKey(0)

#run all frames one time

while(True):
    has_frame, frame = cap.read() # read all frame

    if not has_frame:
        print('End of videos')
        break
    #it convert frames in grayscale
    frame_gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe= cv2.absdiff(frame_gray, gray_median_frame) # difference between gray frame and gray median frame
    #if pixel intensity is greater than the set threshold, value set to 255, else set to 0(Black)
    #th, dframe= cv2.threshold(dframe, 70, 255, cv2.THRESH_BINARY) #IT USED TO CONVERT 0 AND 255 RESPECTIVLY BLACK AND WHITE

    th, dframe= cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #USING  cv2.THRESH.OTSU PROVIDE MINIMUM BINARY VALUE O OR HIGHER AS COMPARE TO MOTION, we don't need value ourself

    print(th)
    cv2.imshow('Frame', dframe) # run to find out what difference in frame and motion detected with background is gary.
    writer.write(dframe) #write all frame

    #cv2.imshow('Frame', frame_gray) #test grayscale frame run in loop

   # cv2.imshow('Frame', frames) # use for showing without gray scale
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
writer.release() # release memory
cap.release() # memory release



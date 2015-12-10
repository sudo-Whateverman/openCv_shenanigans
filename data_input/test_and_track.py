__author__ = 'EL13115'
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
url = 'C:\Documents and Settings\EL13115\Desktop\Megamind.avi'
urleye = 'C:\Documents and Settings\EL13115\My' \
         ' Documents\Downloads\opencv\sources\data\haarcascades\haarcascade_eye.xml'
urlface = 'C:\Documents and Settings\EL13115\My' \
          ' Documents\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml'

# img = cv2.imread(url,0)
# cv2.imshow('image', img)
# k = cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.imshow(img)
# plt.xticks([]), plt.yticks([])
# plt.show()

cap = cv2.VideoCapture(url)
print os.path.exists(url)
print os.path.exists(urleye)
print cap.isOpened()


face_cascade = cv2.CascadeClassifier(urlface)
eye_cascade = cv2.CascadeClassifier(urleye)

### Initial Find Face ###
ret, frame = cap.read()
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

### first find of Face ###
roi_list = []
while ret:
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces):
        for (x, y, w, h) in faces:
            roi_list.append(frame[x:x+w, y:y+h])
        break
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# while True:
#     start = time.time()
#     ret, frame = cap.read()
#     if ret:
#         gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray,1.3,5)
#         for (x, y, w, h) in faces:
#             frame = cv2.rectangle(frame, (x ,y), (x+w, y+h), (255, 0 , 0), 2)
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_hsv = frame[y:y+h, x:x+w]
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex ,ey), (ex+ew, ey+eh), (0, 255 , 0), 2)
#         end = time.time()
#         fps = 1.0/(end-start)
#         cv2.putText(frame,"fps : {0:.2f}".format(fps), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         print("No cam")
#         break



# set up the ROI for tracking
track_window = tuple(faces[0])
roi = roi_list[0]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break

cv2.destroyAllWindows()
cap.release()



__author__ = 'EL13115'
import os
import time

import cv2


class main_args():
    def __init__(self):
        self.url = 'C:\Documents and Settings\EL13115\Desktop\Megamind.avi'
        self.urleye = 'C:\Documents and Settings\EL13115\My' \
                      ' Documents\Downloads\opencv\sources\data\haarcascades\haarcascade_eye.xml'
        self.urlface = 'C:\Documents and Settings\EL13115\My' \
                       ' Documents\Downloads\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml'


def main(init_class):
    # img = cv2.imread(url,0)
    # cv2.imshow('image', img)
    # k = cv2.waitKey(0)
    # if k==27:
    #     cv2.destroyAllWindows()

    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    cap = cv2.VideoCapture(init_class.url)
    print os.path.exists(init_class.url)
    print os.path.exists(init_class.urleye)
    print cap.isOpened()

    face_cascade = cv2.CascadeClassifier(init_class.urlface)
    eye_cascade = cv2.CascadeClassifier(init_class.urleye)

    while True:
        start = time.time()
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            end = time.time()
            fps = 1.0 / (end - start)
            cv2.putText(frame, "fps : {0:.2f}".format(fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No cam")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    help_class = main_args()
    main(help_class)

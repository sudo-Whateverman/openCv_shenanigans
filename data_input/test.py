__author__ = 'EL13115'
import os
import time
import sys
import cv2


class main_args():
    def __init__(self):
        self.urlface = '/home/nick/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml'
        self.urleye = '/home/nick/opencv/data/haarcascades_cuda/haarcascade_eye.xml'


def main(init_class):
    cap = cv2.VideoCapture(0)
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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
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

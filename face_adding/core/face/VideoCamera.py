import cv2
import threading
from face_adding.core.face import classifier2


class VideoCamera(object):
    def __init__(self, le, clf, font):
        self.le = le
        self.clf = clf
        self.font = font
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
        print("ini thread")
        self.isAlive = True

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        classifier2.detect(image, self.le, self.clf, self.font)
        ret, jpeg = cv2.imencode('.jpg', image)
        if not self.isAlive:
            exit()
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

    # @staticmethod
    def stream(self):
        while True:
            frame = self.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

import threading
from face_adding.core.face import classifier2


class MyThread(threading.Thread):
    def run(self):
        classifier2.infer()

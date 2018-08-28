import time

import cv2
import face_recognition
import os

imageDataPath = os.path.join(os.path.expanduser('~'), 'upload', 'training-images')


def video_face_recognition(storePath, baseName):
    count_frame_name = 0
    frame_count = 0
    fps = 0
    start_time = time.time()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    video_capture = cv2.VideoCapture("/home/quypn/IProject/face_recognition/video/therock.mp4")
    while video_capture.isOpened():
        frame_count += 1
        count_frame_name += 1
        det, frame = video_capture.read()
        face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
        print("I found {} face(s) in this photograph.".format(len(face_locations)))
        for face_location in face_locations:
            top, right, bottom, left = face_location
            print(
                "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                      right))
            # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # cv2.imwrite(os.path.join(storePath, "{}{}.jpg".format(baseName, count_frame_name)), frame)
        if time.time() - start_time > 1:
            fps = float("{0:.3}".format(frame_count / (time.time() - start_time)))
            frame_count = 0
            start_time = time.time()

        print("FPS: ", fps)
        cv2.putText(frame, "FPS: {}".format(str(fps)), (5, 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    name = str(input())
    storePath = os.path.join(imageDataPath, name)
    if not os.path.exists(storePath):
        os.makedirs(storePath)
    video_face_recognition(storePath, name)

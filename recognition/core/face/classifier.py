import time

start = time.time()
import argparse
import os
from operator import itemgetter

import numpy as np
import openface
import sys

np.set_printoptions(precision=2)
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import cv2
from sklearn.mixture import GMM
import face_recognition
from PIL import Image

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg, cv2, multiple=False, ):
    start = time.time()
    # bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if myArgs.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if myArgs.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        return None
        # raise Exception("Unable to find a face: {}".format(imgPath))
    if myArgs.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:

        cv2.rectangle(bgrImg, (bb.left(), bb.top()), (bb.right(), bb.bottom()), (0, 0, 255), 2)
        start = time.time()
        alignedFace = align.align(
            myArgs.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            return None
            # raise Exception("Unable to align image: {}".format(imgPath))
        if myArgs.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        if myArgs.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def infer(args, multiple=False):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    frame_count = 0
    fps = 0
    start_time = time.time()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    video_capture = cv2.VideoCapture("/home/quypn/IProject/face_recognition/data/mcem0_head.mpg")
    while video_capture.isOpened():
        det, img = video_capture.read()
        frame_count += 1
        # print("\n=== {} ===".format(img))
        reps = getRep(img, cv2, multiple)
        if reps is not None:
            if len(reps) > 1:
                print("List of faces in image from left to right")
            for r in reps:
                print("rep: ", r[1])
                print("rep_0: ", r[0])
                rep = r[1].reshape(1, -1)
                bbx = r[0]
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                if args.verbose:
                    print("Prediction took {} seconds.".format(time.time() - start))
                if multiple:
                    print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                             confidence))
                else:
                    print("Predict {} with {:.2f} confidence.".format(person, confidence))
                if isinstance(clf, GMM):
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))

        if time.time() - start_time > 1:
            fps = float("{0:.3}".format(frame_count / (time.time() - start_time)))
            frame_count = 0
            start_time = time.time()
        print("FPS: ", fps)
        cv2.putText(img, "FPS: {}".format(str(fps)), (5, 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def train(args):
    global clf
    fname = "{}/labels.csv".format(args.workDir)
    labels = pd.read_csv(fname, header=None).values[:, 1]
    print("labels: ", labels)
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    labels = list(labels)

    # get reps
    fname = "{}/reps.csv".format(args.workDir)
    embeddings = pd.read_csv(fname, header=None).values

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(embeddings, labelsNum)
    print(clf)
    fName = "{}/classifier.pkl".format(args.workDir)
    print("fName", fName)
    print("Saving classifier to '{}'".format(fName))

    f = open(fName, "wb")
    pickle.dump((le, clf), f)


if __name__ == '__main__':
    myParser = argparse.ArgumentParser()
    myParser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                          default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

    myParser.add_argument("--networkModel", type=str, help="Path to Torch networkModel.",
                          default=os.path.join(openfaceModelDir, "nn4.small2.v1.t7"))
    myParser.add_argument('--imgDim', type=int, help="Default image dimention", default=96)

    # todo
    myParser.add_argument("--cuda", action="store_true", default=True)
    myParser.add_argument("-verbose", action="store_true", default=False)

    myParser.add_argument('--classifier', type=str,
                          choices=['LinearSvm', 'GridSearchSvm', 'GMM', 'RadialSvm', 'DecisionTree,GaussianNB',
                                   'DBN'], help='The type of classifier to ues.', default="LinearSvm")
    myParser.add_argument('--workDir', type=str, help='Path to store Model,csv file',
                          default='./data/generated-embeddings')
    myParser.add_argument('--classifierModel', type=str,
                          default='./data/generated-embeddings/classifier.pkl')
    myArgs = myParser.parse_args()

    start = time.time()

    align = openface.AlignDlib(myArgs.dlibFacePredictor)
    net = openface.TorchNeuralNet(myArgs.networkModel, imgDim=myArgs.imgDim,
                                  cuda=myArgs.cuda)

    # train(myArgs)
    # infer(myArgs)

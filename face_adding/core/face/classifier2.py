import argparse
import sys
import time
from operator import itemgetter
import openface
import dlib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from face_adding.core.face.generateRepresentations import *

np.set_printoptions(precision=2)
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import cv2
from sklearn.mixture import GMM
from sklearn.grid_search import GridSearchCV
import face_recognition
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from face_adding.core.face.config.config import Config

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '../augimg', '../models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

pathGenerateRep = os.path.join(os.path.expanduser('~'), 'upload', 'generated-embeddings')
classifierModel = os.path.join(pathGenerateRep, 'classifier.pkl')


def detect(args, img, le, multiple, clf, font, align, net):
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")
    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for face_location in face_locations:
        top, right, bottom, left = face_location

        bb = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        rep = net.forward(alignedFace)
        rep = rep.reshape(1, -1)
        bbx = bb.center().x
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if multiple:
            print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx, confidence))
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            if confidence > Config.threshold:
                cv2.putText(img, "{} {:.2f}%".format(person, confidence), (left, bottom + 20), font, 1, (0, 0, 255), 1,
                            cv2.LINE_AA)
            else:
                cv2.putText(img, "unknown", (left, bottom + 20), font, 1, (0, 0, 255), 1,
                            cv2.LINE_AA)
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))


def infer(args, align, net, multiple=True):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    frame_count = 0
    fps = 0
    start_time = time.time()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        det, img = video_capture.read()
        frame_count += 1

        detect(args, img, le, multiple, clf, font, align, net)

        if time.time() - start_time > 1:
            fps = float("{0:.3}".format(frame_count / (time.time() - start_time)))
            frame_count = 0
            start_time = time.time()

        cv2.putText(img, "FPS: {}".format(str(fps)), (5, 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def train(args):
    # get labels
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

    elif args.classifier == 'GirdSearchSvm':
        print("""
                Warning: In our experiences, using a grid search over SVM hyper-parameters only
                gives marginally better performance than a linear SVM with C=1 and
                is not worth the extra computations of performing a grid search.
                """)
        param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=2)
    elif args.classifier == 'GMM':  # Doesn't work best
        print("user: ", GMM)
        clf = GMM(n_components=nClasses)
    elif args.classifier == 'RadialSvm':
        # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
    elif args.classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
    elif args.classifier == 'GaussianNB':
        clf = GaussianNB()

    if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

    clf.fit(embeddings, labelsNum)
    fName = "{}/classifier.pkl".format(args.workDir)
    print("fName", fName)
    print("Saving classifier to '{}'".format(fName))

    f = open(fName, "wb")
    pickle.dump((le, clf), f)


def init():
    myParser = argparse.ArgumentParser()
    myParser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                          default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

    myParser.add_argument("--networkModel", type=str, help="Path to Torch networkModel.",
                          default=os.path.join(openfaceModelDir, "nn4.small2.v1.t7"))
    myParser.add_argument('--imgDim', type=int, help="Default image dimention", default=96)

    # todo
    myParser.add_argument("--cuda", action="store_true", default=True)
    # myParser.add_argument("--verbose", action="store_true", default=False)

    myParser.add_argument('--classifier', type=str,
                          choices=['LinearSvm',
                                   'GridSearchSvm',
                                   'GMM',
                                   'RadialSvm',
                                   'DecisionTree',
                                   'GaussianNB'], help='The type of classifier to ues.', default="LinearSvm")
    myParser.add_argument('--workDir', type=str, help='Path to store Model,csv file',
                          default=pathGenerateRep)
    myParser.add_argument('--classifierModel', type=str,
                          default=classifierModel)
    myParser.add_argument('--ldaDim', type=int, default=-1)
    myArgs = myParser.parse_args()

    align = openface.AlignDlib(myArgs.dlibFacePredictor)
    net = openface.TorchNeuralNet(myArgs.networkModel, imgDim=myArgs.imgDim,
                                  cuda=myArgs.cuda)
    return myArgs, align, net


if __name__ == '__main__':
    args, align, net = init()
    infer(args, align, net)
    # doGeneratePres()
    train(args)

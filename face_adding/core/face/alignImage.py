import os
import cv2
import openface
from face_adding.models import *

openfaceDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
modelDir = os.path.join(openfaceDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
imageDataPath = os.path.join(os.path.expanduser('~'), 'upload', 'training-images')
dlibFacePredictor = os.path.join(dlibModelDir,
                                 "shape_predictor_68_face_landmarks.dat")
model = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(model, imgDim=imgDim)

pathAlignedImage = os.path.join(os.path.expanduser('~'), 'upload', 'aligned-images')


class AlignImage(object):
    @staticmethod
    def make(name):
        try:
            person = Person.objects.filter(name=name)
            person = person.last()
            store_path = os.path.join(pathAlignedImage, person.name.replace(' ', ''))
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            images = Image.objects.filter(person_id=person.id)
            for image in images:
                print("path: ", image.path)
                bgrImg = cv2.imread(image.path)
                bb = align.getLargestFaceBoundingBox(bgrImg)
                alignedFace = align.align(imgDim, bgrImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                store_align_path = os.path.join(store_path, os.path.basename(image.path))
                cv2.imwrite(store_align_path, alignedFace)

                alignedImg = ImageAligned(path=store_align_path, image_id=image.id)
                alignedImg.save()

                image.aligned = True
                image.save()

        except AttributeError:
            print("Person not found!")

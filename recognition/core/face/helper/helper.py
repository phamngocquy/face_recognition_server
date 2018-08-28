from face.alignImage import loadTreeImage
import os
import numpy as np

from face.config.config import Config


def deleteDatFile():
    arr_image_each = []
    rootPath = os.listdir(Config.trainingImagePath)
    for path in rootPath:
        pathEach = os.path.join(Config.trainingImagePath, path)
        pathImages = os.listdir(pathEach)
        arr_image_each += [pathEach + os.sep + i for i in pathImages]
    for i in arr_image_each:
        if i.endswith(".dat"):
            os.remove(i)


if __name__ == '__main__':
    deleteDatFile()

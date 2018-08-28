import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import cv2
import os
from datetime import datetime
from face_adding.utils.config import Config
from face_adding.models import *


class ImgAugment(object):
    @staticmethod
    def make(path, name):
        person = Person.objects.filter(name=name)
        if len(person) <= 0:
            person = Person(name=name)
            person.save()
        else:
            person = person.last()
        img_raw = Image(path=path, person_id=person.id)
        img_raw.save()

        try:
            img = cv2.imread(path)
            images = np.array(
                [img for _ in range(12)],
                dtype=np.uint8
            )
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(0.0, 30),
                    translate_px=iap.RandomSign(iap.Poisson(3))  # set seed for randomSign
                )
            ])
            images_aug = seq.augment_images(images)
            store_path = os.path.join(Config.storePath, person.name)
            for index, img in enumerate(images_aug):
                img_path = "{}/{}.jpg".format(store_path,
                                              person.name.replace(' ', '') + str(datetime.now().microsecond))
                cv2.imwrite(img_path, img)
                img = Image(path=img_path, person_id=person.id)
                img.save()
        except IOError:
            print("Path not exists!")

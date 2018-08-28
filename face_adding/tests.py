import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_recognition_server.settings")
django.setup()
from django.test import TestCase
from face_adding.models import *
from recognition.core.face.alignImage import *

# Create your tests here.
if __name__ == '__main__':
    # person = Person(name="Quy")
    # person.save()
    # print(person.id)
    # img = Image(path="/home/quypn/IProject/face_recognition/face/data/training-images/jack/jack.jpg",person=person)
    # img.save()
    # images = Image.objects.filter(person_id=7)
    # for i in images:
    #     print(i.path)
    AlignImage.make("Phạm Ngọc")

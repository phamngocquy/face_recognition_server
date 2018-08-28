from face_adding.models import Image, ImageAligned, models
from django.test import TestCase
import os

if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "/home/quypn/IProject/face_recognition_server.settings")
    Image.objects.all()

# class TestDao(TestCase):
#     def getAllOb(self):
#         os.environ.setdefault("DJANGO_SETTINGS_MODULE", "/home/quypn/IProject/face_recognition_server.settings")
#
#         arr = Image.objects.all()
#         print("len: ", len(arr))

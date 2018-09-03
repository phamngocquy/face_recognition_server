from django.http import HttpResponse, StreamingHttpResponse, HttpResponseRedirect
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

from .forms import UploadFileForm
from face_adding.core.augimg.augment import *
from face_adding.core.face.alignImage import *
from face_adding.core.face import classifier2
from face_adding.core.face.VideoCamera import VideoCamera
import os

le, clf, font = classifier2.getClf()
cam = VideoCamera(le, clf, font)


# Create your views here.
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['picture'], request.POST.get('name'))
            return HttpResponse("<h2>Upload successful</h2>")
        else:
            return HttpResponse("<h2>Upload not successful</h2>")
    else:
        form = UploadFileForm()
        persons = Person.objects.all()
    return render(request, '../templates/face_adding/home.html', {'persons': persons})


def handle_uploaded_file(img, name):
    store_path = os.path.join(Config.storePath, replaceSpace(name))
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    image_path = os.path.join(Config.storePath, replaceSpace(name), img.name)

    with open(image_path, "wb+") as file:
        for chunk in img.chunks():
            file.write(chunk)

    # make augment
    ImgAugment.make(image_path, name)

    # align image of person
    AlignImage.make(name)


def replaceSpace(s):
    return s.replace(' ', '')


def video_stream(request):
    return StreamingHttpResponse(cam.stream(), content_type='multipart/x-mixed-replace;boundary=frame')


@api_view(["POST"])
def make_train(request):
    try:
        if request.method == 'POST':
            classifier2.make_training()
    except RuntimeError:
        return Response(data="error when training", status=status.HTTP_400_BAD_REQUEST)
    return Response(data="training completely", status=status.HTTP_200_OK)

# @api_view(["POST"])
# def make_infer(request):
#     try:
#         if request.method == 'POST':
#             if not cam.isAlive:
#                 return streamer
#     except RuntimeError:
#         return Response(status=status.HTTP_409_CONFLICT)

# def infer(request):
#     if request.method == 'GET':
#         if classifier2.FLAG_EXIT == -1:
#             classifier2.FLAG_EXIT = 0
#             classifier2.infer()
#         else:
#             print("infer running")
#     return render(request, '../templates/face_adding/home.html')

# def train(request):
#     if request.method == 'GET':
#         cam.isAlive = False
#         classifier2.make_training()
#     return render(request, '../templates/face_adding/home.html')

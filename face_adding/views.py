from django.http import HttpResponse
from django.shortcuts import render
from .forms import UploadFileForm
from face_adding.core.augimg.augment import *
from face_adding.core.face.alignImage import *
from face_adding.core.face import classifier2

import os
from face_adding.core.thread.MyThread import MyThread

thread = MyThread()


# Create your views here.
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'], request.POST.get('name'))
            return HttpResponse("<h2>Upload successful</h2>")
        else:
            return HttpResponse("<h2>Upload not successful</h2>")
    else:
        form = UploadFileForm()
    return render(request, '../templates/face_adding/home.html', {'form': form})


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


def infer(request):
    if request.method == 'GET':
        if classifier2.FLAG_EXIT == -1:
            classifier2.FLAG_EXIT = 0
            classifier2.infer()
        else:
            print("infer running")
    return render(request, '../templates/face_adding/home.html')


def train(request):
    if request.method == 'GET':
        if classifier2.FLAG_EXIT == 0:
            classifier2.FLAG_EXIT = -1
            classifier2.make_training()
            print("training complete")
            print("start infer")
            classifier2.FLAG_EXIT = 0
            classifier2.infer()
        else:
            classifier2.make_training()
    return render(request, '../templates/face_adding/home.html')

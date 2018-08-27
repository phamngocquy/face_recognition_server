from django.http import HttpResponse
from django.shortcuts import render
from .forms import UploadFileForm
from django.template import loader
from face_adding.utils.config import Config
import os


# Create your views here.
def index(request):
    return render(request, '../templates/face_adding/home.html')


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'], request.POST.get('title'))
            return HttpResponse("<h2>Upload successful</h2>")
        else:
            return HttpResponse("<h2>Upload not successful</h2>")
    else:
        form = UploadFileForm()
    return render(request, '../templates/face_adding/home.html', {'form': form})


def handle_uploaded_file(f, t):
    store_path = os.path.join(Config.storePath, t)

    if not os.path.exists(store_path):
        os.makedirs(store_path)
    image_path = os.path.join(Config.storePath, t, f.name)
    with open(image_path, "wb+") as file:
        for chunk in f.chunks():
            file.write(chunk)

    # make augment

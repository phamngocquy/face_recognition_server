from django.http import HttpResponse
from django.shortcuts import render
from .forms import UploadFileForm
from django.template import loader


# Create your views here.
def index(request):
    return render(request, '../templates/face_adding/home.html')


def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            print("title: ", request.POST.get('title'))
            handle_uploaded_file(request.FILES['file'])
            return HttpResponse("<h2>Upload successful</h2>")
        else:
            return HttpResponse("<h2>Upload not successful</h2>")
    else:
        form = UploadFileForm()
    return render(request, '../templates/face_adding/home.html', {'form': form})


def handle_uploaded_file(f):
    print("name: ", f.name)
    with open(f.name, "wb+") as file:
        for chunk in f.chunks():
            file.write(chunk)

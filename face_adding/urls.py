from django.urls import path
from . import views

urlpatterns = [
    path('add', views.upload_file, name='uploadfile')
]

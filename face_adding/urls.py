from django.urls import path
from . import views

urlpatterns = [
    path('add', views.upload_file, name='uploadfile'),
    path('infer', views.infer, name='infer'),
    path('train', views.train, name='train')
]

from django.urls import path
from . import views

urlpatterns = [
    # path('', views.index, name='index')
    path('', views.upload_file, name='uploadfile')
]

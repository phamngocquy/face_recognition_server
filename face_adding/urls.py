from django.urls import path
from . import views

urlpatterns = [
    path('home', views.upload_file, name='uploadfile'),
    # path('infer', views.make_infer, name='infer'),
    path('train', views.make_train, name='train'),
    path('stream', views.video_stream, name='video-feed')
]

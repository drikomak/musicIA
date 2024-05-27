from django.urls import path
from . import views

import home

urlpatterns = [
    path('', views.index),
    path('api/', views.api),
    path('cam/', views.camera),
    path('api/get_audio_files', views.get_audio_files, name='get_audio_files'),
]
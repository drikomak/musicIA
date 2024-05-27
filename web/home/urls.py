from django.urls import path
from . import views

import home

urlpatterns = [
    path('', views.index),
    path('api/', views.api),
    path('cam/', views.camera),
    path('categ/', views.categ),
]
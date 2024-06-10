from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('api/', views.api),
    path('cam/', views.camera),
    path('setupcam/', views.getCamera),
    path('categ/', views.categ),
]
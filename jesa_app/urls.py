from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze_project/', views.analyze_project, name='analyze_project'),
]
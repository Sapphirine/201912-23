from django.conf.urls import url
from django.urls import path
from . import view
 
urlpatterns = [
    path('dashboard', view.dashboard),
]
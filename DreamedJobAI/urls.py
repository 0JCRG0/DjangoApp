from django.urls import path
from . import views

"""app_name = 'DreamedJobAI'
urlpatterns = [
    path('', views.index, name='index')
]"""

app_name = 'DreamedJobAI'
urlpatterns = [
    path('', views.landing_page, name='landing_page'),
    path('index/', views.index, name='index'),
]
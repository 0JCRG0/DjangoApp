from django.urls import path
from . import views

"""app_name = 'DreamedJobAI'
urlpatterns = [
    path('', views.index, name='index')
]"""

app_name = 'DreamedJobAI'
urlpatterns = [
    path('', views.home, name='home'),
    path('submit_pdf/', views.submit_pdf, name='submit_pdf'),
]
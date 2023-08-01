from django.urls import path
from . import views
from .views import MyLoginView, RegisterView
from django.contrib.auth.views import LogoutView 


app_name = 'DreamedJobAI'
urlpatterns = [
    path('', views.home, name='home'),
    path('submit_pdf/', views.submit_pdf, name='submit_pdf'),
    path('login/', MyLoginView.as_view(template_name='DreamedJobAI/registration/login.html'),name='login'),
    path('logout/', LogoutView.as_view(next_page='/DreamedJobAI/'),name='logout'),
    path('register/', RegisterView.as_view(),name='register')
]
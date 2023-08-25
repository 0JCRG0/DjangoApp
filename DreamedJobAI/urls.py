from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .views import MyLoginView, RegisterView, LegalViews, SidebarViews, ProfileView, InitialJobRequestView, UserJobsView
from django.contrib.auth.views import (
    LogoutView,
    PasswordResetView, 
    PasswordResetDoneView, 
    PasswordResetConfirmView,
    PasswordResetCompleteView
) 


app_name = 'DreamedJobAI'
urlpatterns = [
    path('', views.home, name='home'),
    #path('submit_pdf/', views.submit_pdf, name='submit_pdf'),
    path('login/', MyLoginView.as_view(template_name='DreamedJobAI/registration/login.html'),name='login'),
    path('logout/', LogoutView.as_view(next_page="/"),name='logout'),
    path('register/', RegisterView.as_view(),name='register'),
    path('password-reset/', PasswordResetView.as_view(template_name='DreamedJobAI/password_reset/password_reset.html', email_template_name='DreamedJobAI/password_reset/password_reset_email.html', from_email="maddy@rolehounds.com", success_url='done'),name='password-reset'),
    path('password-reset/done/', PasswordResetDoneView.as_view(template_name='DreamedJobAI/password_reset/password_reset_done.html'),name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', PasswordResetConfirmView.as_view(template_name='DreamedJobAI/password_reset/password_reset_confirm.html', success_url='done'),name='password_reset_confirm'),
    path('password-reset-confirm/<uidb64>/<token>/done',PasswordResetCompleteView.as_view(template_name='DreamedJobAI/password_reset/password_reset_complete.html'),name='password_reset_complete'),
    path('terms-and-conditions/', LegalViews.as_view(), name='TandC'),
    path('privacy-notice/', LegalViews.as_view(), name='privacy-notice'),
    path('home-user/', SidebarViews.as_view(), name='home-user'),
    path('calendar-user/', SidebarViews.as_view(), name='calendar-user'),
    path('request-jobs/', InitialJobRequestView.as_view(), name='request-jobs'),
    path('jobs-user/', UserJobsView.as_view(), name='jobs-user'),
    path('profile-user/', ProfileView.as_view(), name='profile-user')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
from django import forms
from .models import UserText
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User



class UploadPDFForm(forms.ModelForm):
    class Meta:
        model = UserText
        fields = ['pdf_file']

class RegisterForm(UserCreationForm):
    email = forms.EmailField(max_length=254)

    class Meta:
        model = User
        fields = ('username',  'email', 'password1', 'password2', )
from django import forms
from .models import UserText, Profile
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User



class UploadPDFForm(forms.ModelForm):
    class Meta:
        model = UserText
        fields = ['pdf_file']

class RegisterForm(UserCreationForm):
    email = forms.EmailField(max_length=254)
    accept_terms = forms.BooleanField(required=False, widget=forms.CheckboxInput(), label="Have you read our T&C and Privacy Notice?")  # Explicitly set it to render a checkbox

    def clean(self):
        cleaned_data = super().clean()
        accept_terms = cleaned_data.get('accept_terms')

        if not accept_terms:
            self.add_error('accept_terms', 'You must accept the terms and conditions to proceed.')
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name',  'email', 'password1', 'password2', )

class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = [
            'picture',
            'full_name',
            'contact_number',
            'video_conference',
            'linkedin',
            'github',
            'website',
            'messages',
            'other'
        ]


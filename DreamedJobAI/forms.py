from django import forms
from .models import Profile, UserCV, UserProfilePreferences
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class UserCVForm(forms.ModelForm):
    class Meta:
        model = UserCV
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
            'contact_number',
            'linkedin',
            'github',
            'website',
            'other'
        ]
        widgets = {
            'country': forms.Select(attrs={'class': 'form-control'}),
        }

class UserProfilePreferencesForm(forms.ModelForm):
    class Meta:
        model = UserProfilePreferences
        fields = [
            'about',
            'desired_job_title',
            'desired_country',
            'second_desired_country',
            'desired_location',
            'desired_job_description',
            'desired_compensation',
            'desired_benefits',
            'desired_industry',
            'desired_start_day',
            'urgency',
        ]
        widgets = {
            'desired_country': forms.Select(attrs={'class': 'form-control'}),
            'second_desired_country': forms.Select(attrs={'class': 'form-control'}),
            'desired_location': forms.Select(attrs={'class': 'form-control'}),
            'desired_compensation': forms.Select(attrs={'class': 'form-control'}),
            'desired_industry': forms.Select(attrs={'class': 'form-control'}),
            'desired_start_day': forms.Select(attrs={'class': 'form-control'}),
            'urgency': forms.Select(attrs={'class': 'form-control'}),
        }

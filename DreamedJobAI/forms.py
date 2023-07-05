from django import forms
from .models import UserText


class UploadPDFForm(forms.ModelForm):
    class Meta:
        model = UserText
        fields = ['pdf_file']

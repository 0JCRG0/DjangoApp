from django.shortcuts import render, redirect
from .forms import UploadPDFForm
from .models import UserText
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from django.core.files.base import ContentFile
import magic
from .utils import summarise_cv, extract_text_from_pdf

def home(request):
    return render(request, 'DreamedJobAI/home.html')

def index(request):
    if request.method == 'POST':
        form = UploadPDFForm(request.POST, request.FILES)
        if form.is_valid():
            user_text = form.save(commit=False)
            pdf_file = request.FILES['pdf_file']
            user_text.pdf_file = pdf_file
            extracted_text = extract_text_from_pdf(pdf_file)
            user_text.extracted_text = extracted_text
            user_text.save()
            summarised_cv = summarise_cv(extracted_text)
            return render(request, 'DreamedJobAI/summary.html', {'summarised_cv': summarised_cv})
    else:
        form = UploadPDFForm()
    return render(request, 'DreamedJobAI/index.html', {'form': form})


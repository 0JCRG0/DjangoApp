from django.shortcuts import render, redirect
from .forms import UploadPDFForm
from .models import UserText
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from django.core.files.base import ContentFile
import magic

def index(request):
    if request.method == 'POST':
        form = UploadPDFForm(request.POST, request.FILES)
        if form.is_valid():
            user_text = form.save(commit=False)
            pdf_file = request.FILES['pdf_file']
            user_text.pdf_file = pdf_file
            user_text.extracted_text = extract_text_from_pdf(pdf_file)
            user_text.save()
            return redirect('DreamedJobAI:index')
    else:
        form = UploadPDFForm()
    return render(request, 'DreamedJobAI/index.html', {'form': form})

def extract_text_from_pdf(pdf_file):
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(pdf_file.read())
        if mime_type == 'application/pdf':
            pdf_file.seek(0)
            resource_manager = PDFResourceManager()
            text_stream = io.StringIO()
            device = TextConverter(resource_manager, text_stream)
            interpreter = PDFPageInterpreter(resource_manager, device)

            for page in PDFPage.get_pages(pdf_file):
                interpreter.process_page(page)

            extracted_text = text_stream.getvalue()
            device.close()
            text_stream.close()
            return extracted_text
        else:
            return None
    except Exception:
        return None

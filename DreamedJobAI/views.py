from .forms import UploadPDFForm
from .utils import summarise_cv, extract_text_from_pdf
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.contrib import messages
from django.http import HttpRequest
from django.core.mail import send_mail
from django.contrib.auth import login
from django.shortcuts import render
from django.views.generic.edit import FormView
from .forms import RegisterForm



def home(request: HttpRequest):
    return render(request, 'DreamedJobAI/home.html')

def submit_pdf(request: HttpRequest):
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
            return render(request, 'DreamedJobAI/summarise.html', {'summarised_cv': summarised_cv})
    else:
        form = UploadPDFForm()
    return render(request, 'DreamedJobAI/submit_pdf.html', {'form': form})


class MyLoginView(LoginView):
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('DreamedJobAI:submit_pdf') 
    
    def form_invalid(self, form):
        messages.error(self.request,'Invalid username or password')
        return self.render_to_response(self.get_context_data(form=form))

class RegisterView(FormView):
    template_name = 'DreamedJobAI/registration/register.html'
    form_class = RegisterForm
    redirect_authenticated_user = True
    success_url = reverse_lazy('DreamedJobAI:submit_pdf')
    
    def form_valid(self, form):
        user = form.save()
        if user:
            login(self.request, user)
            send_mail(
                'Welcome to RoleHounds',
                f'Thank you for registering to RoleHounds! Your now ready to login using your username: {user.username}. ',
                'maddy@rolehounds.com',
                [user.email],
                fail_silently=False,
            )
        
        return super(RegisterView, self).form_valid(form)
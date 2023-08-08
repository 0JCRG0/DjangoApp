from .utils import summarise_cv, extract_text_from_pdf
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.contrib import messages
from django.http import HttpRequest
from django.core.mail import send_mail
from django.contrib.auth import login
from django.shortcuts import render, redirect
from django.views.generic.edit import FormView
from .forms import RegisterForm, ProfileForm, UploadPDFForm, ProfilePreferencesForm
from .models import UserText, Profile, ProfilePreferences
from django.views.generic import TemplateView
from django.views import View
from django.shortcuts import get_object_or_404






def home(request: HttpRequest):
    return render(request, 'DreamedJobAI/index.html')

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
            return render(request, 'DreamedJobAI/user/summarise.html', {'summarised_cv': summarised_cv})
    else:
        form = UploadPDFForm()
    return render(request, 'DreamedJobAI/user/submit_pdf.html', {'form': form})


class MyLoginView(LoginView):
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('DreamedJobAI:home-user') 
    
    def form_invalid(self, form):
        messages.error(self.request,'Invalid username or password')
        return self.render_to_response(self.get_context_data(form=form, error_message='Invalid username or password'))

class RegisterView(FormView):
    template_name = 'DreamedJobAI/registration/register.html'
    form_class = RegisterForm
    redirect_authenticated_user = True
    success_url = reverse_lazy('DreamedJobAI:home-user')
    
    def form_valid(self, form):
        user = form.save()
        if user:
            # Create a Profile instance for the new user
            Profile.objects.create(user=user)
            
            login(self.request, user)
            send_mail(
                'Welcome to RoleHounds',
                f'Thank you for registering to RoleHounds! Your now ready to login using your username: {user.username}. ',
                'maddy@rolehounds.com',
                [user.email],
                fail_silently=False,
            )
        
        return super(RegisterView, self).form_valid(form)
    
    def form_invalid(self, form):
        return self.render_to_response(self.get_context_data(form=form))


class LegalViews(TemplateView):
    def get_template_names(self):
        # Check the view name to determine which template to use
        if self.request.path == '/terms-and-conditions/':
            return 'DreamedJobAI/legal/TandC.html'
        elif self.request.path == '/privacy-notice/':
            return 'DreamedJobAI/legal/PrivacyNotice.html'
        else:
            # Default template if the view name doesn't match any of the above
            return 'default_template.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add any additional context data here if needed
        return context

class SidebarViews(TemplateView):
    def get_template_names(self):
        # TODO: Add template view for the rest of the sidebar buttons
        if self.request.path == '/home-user/':
            return 'DreamedJobAI/user/home-user.html'
        elif self.request.path == '/profile-user/':
            return 'DreamedJobAI/user/profile-user.html'
        else:
            return 'default_template.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add any additional context data here if needed
        context['user'] = self.request.user
        context['profile_picture'] = self.request.user.profile.picture.url
        return context

class ProfileView(View):
    template_name = 'DreamedJobAI/user/replace-profile-user.html'

    def get(self, request):
        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_form = ProfileForm(instance=profile)

        profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
        preferences_form = ProfilePreferencesForm(instance=profile_preferences)

        return render(request, self.template_name, {'profile_form': profile_form, 'preferences_form': preferences_form, 'profile': profile})
    
    def post(self, request):
        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)

        if 'country' in request.POST:
            profile_form = ProfileForm(request.POST, request.FILES, instance=profile)
            if profile_form.is_valid():
                profile = profile_form.save(commit=False)
                profile.user = request.user
                profile.save()
        else:
            preferences_form = ProfilePreferencesForm(request.POST, instance=profile_preferences)
            if preferences_form.is_valid():
                profile_preferences = preferences_form.save(commit=False)
                profile_preferences.user = request.user
                profile_preferences.save()

        if profile_form.is_valid() or preferences_form.is_valid():
            return redirect('DreamedJobAI:profile-user')

        return render(request, self.template_name, {'profile_form': profile_form, 'preferences_form': preferences_form, 'profile': profile})

"""
class ProfileView(View):
    template_name = 'DreamedJobAI/user/replace-profile-user.html'

    def get(self, request):
        profile, created = Profile.objects.get_or_create(user=request.user)
        form = ProfileForm(instance=profile)
        return render(request, self.template_name, {'form': form, 'profile': profile})
    
    def post(self, request):
        profile, created = Profile.objects.get_or_create(user=request.user)
        form = ProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            return redirect('DreamedJobAI:profile-user')
        else:
            print(form.errors)
        return render(request, self.template_name, {'form': form})
"""

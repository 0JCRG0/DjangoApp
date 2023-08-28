from .others import summarise_cv, extract_text_from_pdf
from django.contrib.auth.views import LoginView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.http import HttpRequest
from django.http import JsonResponse
from django.core.mail import send_mail
from django.contrib.auth import login
from django.shortcuts import render, redirect
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from django.views.generic.edit import FormView
from .forms import RegisterForm, ProfileForm, ProfilePreferencesForm, UserCVForm
from .models import Profile, ProfilePreferences, UserCV, SuitableJobs
from django.views.generic import TemplateView
import logging
from django.views import View
from django.shortcuts import get_object_or_404
from .gpt4_utils import main, additional_suitable_jobs
import asyncio


def home(request: HttpRequest):
    return render(request, 'DreamedJobAI/index.html')

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

class SidebarViews(LoginRequiredMixin, TemplateView):
    login_url = '/login/'

    def get_template_names(self):
        if self.request.path == '/home-user/':
            return 'DreamedJobAI/user/home-user.html'
        elif self.request.path == '/calendar-user/':
            return 'DreamedJobAI/user/calendar-user.html'
        #elif self.request.path == '/jobs-user/':
            #return 'DreamedJobAI/user/jobs-user.html'
        else:
            return 'default_template.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        if self.request.user.is_authenticated:
            # Only access user's profile if authenticated
            context['user'] = self.request.user
            if hasattr(self.request.user, 'profile'):
                context['profile_picture'] = self.request.user.profile.picture.url
            else:
                context['profile_picture'] = None
        else:
            context['user'] = None
            context['profile_picture'] = None

        return context

class ProfileView(LoginRequiredMixin, View):
    login_url = '/login/'
    template_name = 'DreamedJobAI/user/profile-user.html'

    def get(self, request: HttpRequest):
        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_form = ProfileForm(instance=profile)

        profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
        preferences_form = ProfilePreferencesForm(instance=profile_preferences)

        password_form = PasswordChangeForm(request.user)

        profile_cv, created = UserCV.objects.get_or_create(user=request.user)
        cv_form = UserCVForm(instance=profile_cv)

        return render(request, self.template_name, {'profile_form': profile_form, 'preferences_form': preferences_form, 'password_form': password_form, 'cv_form': cv_form, 'profile': profile, 'profile_preferences': profile_preferences, 'profile_cv': profile_cv})
    
    def post(self, request: HttpRequest):
        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_form = ProfileForm(instance=profile)

        profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
        preferences_form = ProfilePreferencesForm(instance=profile_preferences)

        password_form = PasswordChangeForm(request.user, request.POST)

        profile_cv, created = UserCV.objects.get_or_create(user=request.user)
        cv_form = UserCVForm(instance=profile_cv)

        success_pdf = None
        failure_pdf = None
        success_password = None
        failure_password = None

        if 'picture' in request.POST:
            profile_form = ProfileForm(request.POST, request.FILES, instance=profile)
            if profile_form.is_valid():
                profile = profile_form.save(commit=False)
                profile.user = request.user
                profile.save()
        elif 'desired_location' in request.POST:
            preferences_form = ProfilePreferencesForm(request.POST, instance=profile_preferences)
            if preferences_form.is_valid():
                profile_preferences = preferences_form.save(commit=False)
                profile_preferences.user = request.user
                profile_preferences.save()
        if set(cv_form.fields.keys()).intersection(request.POST.keys()) or set(cv_form.fields.keys()).intersection(request.FILES.keys()):
            cv_form = UserCVForm(request.POST, request.FILES, instance=profile_cv)
            if cv_form.is_valid():
                user_text = cv_form.save(commit=False)
                if 'pdf_file' in request.FILES:
                    pdf_file = request.FILES['pdf_file']
                    user_text.pdf_file = pdf_file
                    extracted_text = extract_text_from_pdf(pdf_file)
                    user_text.extracted_text = extracted_text
                    summary = summarise_cv(extracted_text)
                    user_text.summary = summary
                    user_text.save()
                    success_pdf = "Your CV was successfully uploaded!"
                else:
                    failure_pdf = "Oops! There was an error while submitting your CV. Please ensure your file is in PDF format and try again."
        else:  # Check if the password form is being submitted
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)  # Important, to update the session with the new password
                success_password = 'Your password was successfully updated!'
            else:
                failure_password="Uh-oh! Something went wrong while changing your password. Please check your input and try again."

        if profile_form.is_valid() or preferences_form.is_valid():
            return redirect('DreamedJobAI:profile-user')

        return render(
                        request,
                        self.template_name,
                            {
                            'profile_form': profile_form,
                            'preferences_form': preferences_form,
                            'password_form': password_form,
                            'cv_form': cv_form,
                            'profile': profile,
                            'profile_preferences': profile_preferences,
                            'profile_cv': profile_cv,
                            'success_pdf': success_pdf,
                            'success_password': success_password,
                            'failure_pdf': failure_pdf,
                            'failure_password': failure_password
                            }
                    )

class InitialJobRequestView(LoginRequiredMixin, View):
    login_url = '/login/'
    template_name = "DreamedJobAI/user/request-jobs.html"
    
    def post(self, request):
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            profile, created = Profile.objects.get_or_create(user=request.user)
            profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
            profile_cv, created = UserCV.objects.get_or_create(user=request.user)

            
            # Fetch the user_id and desired_country from the retrieved objects
            USER_ID = profile.user_id
            USER_COUNTRY = profile_preferences.desired_country
            USER_CV = profile_cv.summary

            #Summarise the raw user cv

            if USER_COUNTRY and USER_CV:
                df = asyncio.run(main(user_id=USER_ID, user_country=USER_COUNTRY, user_cv=USER_CV, top_n_interval=4, num_suitable_jobs=1))
            
                # Convert the DataFrame to a dictionary
                df_dict = df.to_dict(orient='records')

                # Save the data to the database
                for item in df_dict:
                    item['job_id'] = item.pop('id')  # This line changes the key 'id' to 'job_id'
                    SuitableJobs.objects.create(**item)

                return JsonResponse({"success": True}, status=200)
            else:
                return JsonResponse({"success": False}, status=400)
        return JsonResponse({"success": False}, status=400)

    def get(self, request:HttpRequest):
        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
        
        user_id = profile.user_id
        desired_country = profile_preferences.desired_country
        profile_picture = profile.picture.url

        # Retrieve the data from the database
        jobs = SuitableJobs.objects.all()

        filtered_jobs = jobs.filter(user_id=user_id)

        context = {
            'user_id': user_id,
            'desired_country': desired_country,
            'profile_picture': profile_picture,
            'jobs': filtered_jobs
        }

        # Pass the context to the template when rendering
        return render(request, self.template_name, context)

class UserJobsView(LoginRequiredMixin, View):
    login_url = '/login/'
    template_name = "DreamedJobAI/user/jobs-user.html"
    
    def post(self, request):
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            profile, created = Profile.objects.get_or_create(user=request.user)
            profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
            profile_cv, created = UserCV.objects.get_or_create(user=request.user)

            
            # Fetch the user_id and desired_country from the retrieved objects
            USER_ID = profile.user_id
            USER_COUNTRY = profile_preferences.desired_country
            USER_CV = profile_cv.summary

            #Summarise the raw user cv

            if USER_COUNTRY and USER_CV:
                df = asyncio.run(additional_suitable_jobs(user_id=USER_ID, user_country=USER_COUNTRY, user_cv=USER_CV, top_n_interval=4, num_suitable_jobs=1))
            
                # Convert the DataFrame to a dictionary
                df_dict = df.to_dict(orient='records')

                # Save the data to the database
                for item in df_dict:
                    item['job_id'] = item.pop('id')  # This line changes the key 'id' to 'job_id'
                    SuitableJobs.objects.create(**item)

                return JsonResponse({"success": True}, status=200)
            else:
                return JsonResponse({"success": False}, status=400)
        return JsonResponse({"success": False}, status=400)

    def get(self, request:HttpRequest):
        profile, created = Profile.objects.get_or_create(user=request.user)
        profile_preferences, created = ProfilePreferences.objects.get_or_create(user=request.user)
        
        user_id = profile.user_id
        desired_country = profile_preferences.desired_country
        profile_picture = profile.picture.url

        # Retrieve the data from the database
        jobs = SuitableJobs.objects.all()

        filtered_jobs = jobs.filter(user_id=user_id)

        context = {
            'user_id': user_id,
            'desired_country': desired_country,
            'profile_picture': profile_picture,
            'jobs': filtered_jobs
        }

        # Pass the context to the template when rendering
        return render(request, self.template_name, context)

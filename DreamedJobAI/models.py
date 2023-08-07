from django.db import models
from django.contrib.auth.models import User

# ----------------------------------------------/
# PDF MODEL---------------------------/
#-----------------------------------------------/

class UserText(models.Model):
    text = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to='pdf_files/', null=True)
    extracted_text = models.TextField(null=True)

    def __str__(self):
        return self.text
    
# ----------------------------------------------/
# Email is unique now---------------------------/
#-----------------------------------------------/

User._meta.get_field('email')._unique = True

# ----------------------------------------------/
# Profile --------------------------------------/
#-----------------------------------------------/

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    picture = models.ImageField(upload_to='DreamedJobAI/static/DreamedJobAI/assets/img/user_pp')
    contact_number = models.CharField(max_length=250)
    video_conference = models.CharField(max_length=250)
    linkedin = models.CharField(max_length=250)
    github = models.CharField(max_length=250)
    website = models.CharField(max_length=250)
    messages = models.CharField(max_length=250)
    other = models.CharField(max_length=250)


    def __str__(self):
        return self.user.username
    
class ProfilePreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    about = models.CharField(max_length=500)
    desired_job_title = models.CharField(max_length=100)
    desired_location = models.CharField(max_length=100)
    desired_job_description = models.CharField(max_length=500)
    desired_compensantion = models.FloatField(blank=True, null=True)
    desired_benefits = models.CharField(max_length=200)
    desired_industry = models.CharField(max_length=50)
    desired_start_day = models.DateTimeField(max_length=50)
    urgency = models.CharField(max_length=50)


    def __str__(self):
        return self.user.username
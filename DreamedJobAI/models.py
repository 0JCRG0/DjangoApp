from django.db import models

# Create your models here.

class UserText(models.Model):
    text = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to='pdf_files/', null=True)
    extracted_text = models.TextField(null=True)

    def __str__(self):
        return self.text
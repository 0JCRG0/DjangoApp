# Generated by Django 4.2.2 on 2023-08-24 15:31

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0026_delete_job"),
    ]

    operations = [
        migrations.AddField(
            model_name="usercv",
            name="summary",
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name="usercv",
            name="pdf_file",
            field=models.FileField(upload_to="pdf_files/"),
        ),
    ]

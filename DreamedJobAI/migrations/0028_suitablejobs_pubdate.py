# Generated by Django 4.2.2 on 2023-08-28 18:52

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0027_usercv_summary_alter_usercv_pdf_file"),
    ]

    operations = [
        migrations.AddField(
            model_name="suitablejobs",
            name="pubdate",
            field=models.DateTimeField(default=django.utils.timezone.now),
            preserve_default=False,
        ),
    ]

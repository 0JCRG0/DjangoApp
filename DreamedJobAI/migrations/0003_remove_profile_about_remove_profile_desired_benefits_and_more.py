# Generated by Django 4.2.2 on 2023-08-05 08:58

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("DreamedJobAI", "0002_profile"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="profile",
            name="about",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_benefits",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_compensantion",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_industry",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_job_description",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_job_title",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_location",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="desired_start_day",
        ),
        migrations.RemoveField(
            model_name="profile",
            name="urgency",
        ),
        migrations.CreateModel(
            name="ProfilePreferences",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "picture",
                    models.ImageField(
                        upload_to="DreamedJobAI/static/DreamedJobAI/assets/img/user_pp"
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("about", models.CharField(max_length=500)),
                ("desired_job_title", models.CharField(max_length=100)),
                ("desired_location", models.CharField(max_length=100)),
                ("desired_job_description", models.CharField(max_length=500)),
                ("desired_compensantion", models.FloatField(blank=True, null=True)),
                ("desired_benefits", models.CharField(max_length=200)),
                ("desired_industry", models.CharField(max_length=50)),
                ("desired_start_day", models.DateTimeField(max_length=50)),
                ("urgency", models.CharField(max_length=50)),
                (
                    "user",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
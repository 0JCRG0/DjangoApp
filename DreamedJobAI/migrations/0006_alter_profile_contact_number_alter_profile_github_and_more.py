# Generated by Django 4.2.2 on 2023-08-07 16:42

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0005_remove_profile_full_name"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="contact_number",
            field=models.CharField(blank=True, max_length=250),
        ),
        migrations.AlterField(
            model_name="profile",
            name="github",
            field=models.CharField(blank=True, max_length=250),
        ),
        migrations.AlterField(
            model_name="profile",
            name="linkedin",
            field=models.CharField(blank=True, max_length=250),
        ),
        migrations.AlterField(
            model_name="profile",
            name="messages",
            field=models.CharField(blank=True, max_length=250),
        ),
        migrations.AlterField(
            model_name="profile",
            name="other",
            field=models.CharField(blank=True, max_length=250),
        ),
        migrations.AlterField(
            model_name="profile",
            name="picture",
            field=models.ImageField(
                default="DreamedJobAI/assets/img/user.png",
                upload_to="DreamedJobAI/static/DreamedJobAI/assets/img/user_pp",
            ),
        ),
        migrations.AlterField(
            model_name="profile",
            name="video_conference",
            field=models.CharField(blank=True, max_length=250),
        ),
        migrations.AlterField(
            model_name="profile",
            name="website",
            field=models.CharField(blank=True, max_length=250),
        ),
    ]

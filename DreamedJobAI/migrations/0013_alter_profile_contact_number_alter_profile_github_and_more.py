# Generated by Django 4.2.2 on 2023-08-08 14:41

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0012_alter_profile_contact_number_alter_profile_country"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="contact_number",
            field=models.CharField(blank=True, default="Not specified"),
        ),
        migrations.AlterField(
            model_name="profile",
            name="github",
            field=models.URLField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name="profile",
            name="linkedin",
            field=models.URLField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name="profile",
            name="website",
            field=models.URLField(blank=True, max_length=100),
        ),
    ]

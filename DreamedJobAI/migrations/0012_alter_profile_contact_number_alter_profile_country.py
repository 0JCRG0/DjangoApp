# Generated by Django 4.2.2 on 2023-08-08 14:32

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0011_alter_profile_contact_number"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="contact_number",
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="profile",
            name="country",
            field=models.CharField(blank=True, default="Not specified", max_length=50),
        ),
    ]

# Generated by Django 4.2.2 on 2023-08-23 18:59

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0024_suitablejobs_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="suitablejobs",
            name="link",
            field=models.URLField(max_length=1000),
        ),
    ]

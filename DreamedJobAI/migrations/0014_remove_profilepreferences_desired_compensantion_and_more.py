# Generated by Django 4.2.2 on 2023-08-08 16:49

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "DreamedJobAI",
            "0013_alter_profile_contact_number_alter_profile_github_and_more",
        ),
    ]

    operations = [
        migrations.RemoveField(
            model_name="profilepreferences",
            name="desired_compensantion",
        ),
        migrations.AddField(
            model_name="profilepreferences",
            name="desired_compensation",
            field=models.CharField(
                blank=True,
                choices=[
                    ("1-10000", "1-10,000"),
                    ("10001-20000", "10,000-20,000"),
                    ("20001-30000", "20,001-30,000"),
                    ("30001-40000", "30,001-40,000"),
                    ("40001-50000", "40,001-50,000"),
                    ("50001-60000", "50,001-60,000"),
                    ("60001-70000", "60,001-70,000"),
                    ("70001-80000", "70,001-80,000"),
                    ("80001-90000", "80,001-90,000"),
                    ("90001-100000", "90,001-100,000"),
                    ("100000+", "100,000+"),
                ],
                max_length=20,
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="profile",
            name="contact_number",
            field=models.CharField(default="Not specified"),
        ),
        migrations.AlterField(
            model_name="profile",
            name="country",
            field=models.CharField(
                choices=[
                    ("afghanistan", "Afghanistan"),
                    ("albania", "Albania"),
                    ("algeria", "Algeria"),
                    ("andorra", "Andorra"),
                    ("angola", "Angola"),
                    ("antigua_barbuda", "Antigua and Barbuda"),
                    ("argentina", "Argentina"),
                    ("armenia", "Armenia"),
                    ("australia", "Australia"),
                    ("austria", "Austria"),
                    ("azerbaijan", "Azerbaijan"),
                    ("bahamas", "Bahamas"),
                    ("bahrain", "Bahrain"),
                    ("bangladesh", "Bangladesh"),
                    ("barbados", "Barbados"),
                    ("belarus", "Belarus"),
                    ("belgium", "Belgium"),
                    ("belize", "Belize"),
                    ("benin", "Benin"),
                    ("bhutan", "Bhutan"),
                    ("bolivia", "Bolivia"),
                    ("bosnia_herzegovina", "Bosnia and Herzegovina"),
                    ("botswana", "Botswana"),
                    ("brazil", "Brazil"),
                    ("brunei", "Brunei"),
                    ("bulgaria", "Bulgaria"),
                    ("burkina_faso", "Burkina Faso"),
                    ("burundi", "Burundi"),
                    ("cabo_verde", "Cabo Verde"),
                    ("cambodia", "Cambodia"),
                    ("cameroon", "Cameroon"),
                    ("canada", "Canada"),
                    ("central_african_republic", "Central African Republic"),
                    ("chad", "Chad"),
                    ("chile", "Chile"),
                    ("china", "China"),
                    ("colombia", "Colombia"),
                    ("comoros", "Comoros"),
                    ("congo", "Congo"),
                    ("costa_rica", "Costa Rica"),
                    ("croatia", "Croatia"),
                    ("cuba", "Cuba"),
                    ("cyprus", "Cyprus"),
                    ("czech_republic", "Czech Republic"),
                    ("denmark", "Denmark"),
                    ("djibouti", "Djibouti"),
                    ("dominica", "Dominica"),
                    ("dominican_republic", "Dominican Republic"),
                    ("ecuador", "Ecuador"),
                    ("egypt", "Egypt"),
                    ("el_salvador", "El Salvador"),
                    ("equatorial_guinea", "Equatorial Guinea"),
                    ("eritrea", "Eritrea"),
                    ("estonia", "Estonia"),
                    ("eswatini", "Eswatini"),
                    ("ethiopia", "Ethiopia"),
                    ("fiji", "Fiji"),
                    ("finland", "Finland"),
                    ("france", "France"),
                    ("gabon", "Gabon"),
                    ("gambia", "Gambia"),
                    ("georgia", "Georgia"),
                    ("germany", "Germany"),
                    ("ghana", "Ghana"),
                    ("greece", "Greece"),
                    ("grenada", "Grenada"),
                    ("guatemala", "Guatemala"),
                    ("guinea", "Guinea"),
                    ("guinea_bissau", "Guinea-Bissau"),
                    ("guyana", "Guyana"),
                    ("haiti", "Haiti"),
                    ("honduras", "Honduras"),
                    ("hungary", "Hungary"),
                    ("iceland", "Iceland"),
                    ("india", "India"),
                    ("indonesia", "Indonesia"),
                    ("iran", "Iran"),
                    ("iraq", "Iraq"),
                    ("ireland", "Ireland"),
                    ("israel", "Israel"),
                    ("italy", "Italy"),
                    ("ivory_coast", "Ivory Coast"),
                    ("jamaica", "Jamaica"),
                    ("japan", "Japan"),
                    ("jordan", "Jordan"),
                    ("kazakhstan", "Kazakhstan"),
                    ("kenya", "Kenya"),
                    ("kiribati", "Kiribati"),
                    ("north_korea", "North Korea"),
                    ("south_korea", "South Korea"),
                    ("kosovo", "Kosovo"),
                    ("kuwait", "Kuwait"),
                    ("kyrgyzstan", "Kyrgyzstan"),
                    ("laos", "Laos"),
                    ("latvia", "Latvia"),
                    ("lebanon", "Lebanon"),
                    ("lesotho", "Lesotho"),
                    ("liberia", "Liberia"),
                    ("libya", "Libya"),
                    ("liechtenstein", "Liechtenstein"),
                    ("lithuania", "Lithuania"),
                    ("luxembourg", "Luxembourg"),
                    ("madagascar", "Madagascar"),
                    ("malawi", "Malawi"),
                    ("malaysia", "Malaysia"),
                    ("maldives", "Maldives"),
                    ("mali", "Mali"),
                    ("malta", "Malta"),
                    ("marshall_islands", "Marshall Islands"),
                    ("mauritania", "Mauritania"),
                    ("mauritius", "Mauritius"),
                    ("mexico", "Mexico"),
                    ("micronesia", "Micronesia"),
                    ("moldova", "Moldova"),
                    ("monaco", "Monaco"),
                    ("mongolia", "Mongolia"),
                    ("montenegro", "Montenegro"),
                    ("morocco", "Morocco"),
                    ("mozambique", "Mozambique"),
                    ("myanmar", "Myanmar"),
                    ("namibia", "Namibia"),
                    ("nauru", "Nauru"),
                    ("nepal", "Nepal"),
                    ("netherlands", "Netherlands"),
                    ("new_zealand", "New Zealand"),
                    ("nicaragua", "Nicaragua"),
                    ("niger", "Niger"),
                    ("nigeria", "Nigeria"),
                    ("north_macedonia", "North Macedonia"),
                    ("norway", "Norway"),
                    ("oman", "Oman"),
                    ("pakistan", "Pakistan"),
                    ("palau", "Palau"),
                    ("panama", "Panama"),
                    ("papua_new_guinea", "Papua New Guinea"),
                    ("paraguay", "Paraguay"),
                    ("peru", "Peru"),
                    ("philippines", "Philippines"),
                    ("poland", "Poland"),
                    ("portugal", "Portugal"),
                    ("qatar", "Qatar"),
                    ("romania", "Romania"),
                    ("russia", "Russia"),
                    ("rwanda", "Rwanda"),
                    ("saint_kitts_nevis", "Saint Kitts and Nevis"),
                    ("saint_lucia", "Saint Lucia"),
                    ("saint_vincent_grenadines", "Saint Vincent and the Grenadines"),
                    ("samoa", "Samoa"),
                    ("san_marino", "San Marino"),
                    ("sao_tome_principe", "Sao Tome and Principe"),
                    ("saudi_arabia", "Saudi Arabia"),
                    ("senegal", "Senegal"),
                    ("serbia", "Serbia"),
                    ("seychelles", "Seychelles"),
                    ("sierra_leone", "Sierra Leone"),
                    ("singapore", "Singapore"),
                    ("slovakia", "Slovakia"),
                    ("slovenia", "Slovenia"),
                    ("solomon_islands", "Solomon Islands"),
                    ("somalia", "Somalia"),
                    ("south_africa", "South Africa"),
                    ("south_sudan", "South Sudan"),
                    ("spain", "Spain"),
                    ("sri_lanka", "Sri Lanka"),
                    ("sudan", "Sudan"),
                    ("suriname", "Suriname"),
                    ("sweden", "Sweden"),
                    ("switzerland", "Switzerland"),
                    ("syria", "Syria"),
                    ("taiwan", "Taiwan"),
                    ("tajikistan", "Tajikistan"),
                    ("tanzania", "Tanzania"),
                    ("thailand", "Thailand"),
                    ("timor_leste", "Timor-Leste"),
                    ("togo", "Togo"),
                    ("tonga", "Tonga"),
                    ("trinidad_tobago", "Trinidad and Tobago"),
                    ("tunisia", "Tunisia"),
                    ("turkey", "Turkey"),
                    ("turkmenistan", "Turkmenistan"),
                    ("tuvalu", "Tuvalu"),
                    ("uganda", "Uganda"),
                    ("ukraine", "Ukraine"),
                    ("united_arab_emirates", "United Arab Emirates"),
                    ("united_kingdom", "United Kingdom"),
                    ("united_states", "United States"),
                    ("uruguay", "Uruguay"),
                    ("uzbekistan", "Uzbekistan"),
                    ("vanuatu", "Vanuatu"),
                    ("vatican_city", "Vatican City"),
                    ("venezuela", "Venezuela"),
                    ("vietnam", "Vietnam"),
                    ("yemen", "Yemen"),
                    ("zambia", "Zambia"),
                    ("zimbabwe", "Zimbabwe"),
                ],
                max_length=35,
            ),
        ),
        migrations.AlterField(
            model_name="profile",
            name="github",
            field=models.CharField(default="Not specified", max_length=100),
        ),
        migrations.AlterField(
            model_name="profile",
            name="linkedin",
            field=models.CharField(default="Not specified", max_length=100),
        ),
        migrations.AlterField(
            model_name="profile",
            name="other",
            field=models.CharField(default="Not specified", max_length=100),
        ),
        migrations.AlterField(
            model_name="profile",
            name="state",
            field=models.CharField(default="Not specified", max_length=50),
        ),
        migrations.AlterField(
            model_name="profile",
            name="website",
            field=models.CharField(default="Not specified", max_length=100),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="about",
            field=models.TextField(default="Not specified"),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="desired_benefits",
            field=models.CharField(default="Not specified", max_length=200),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="desired_industry",
            field=models.CharField(
                blank=True,
                choices=[
                    ("technology", "Technology"),
                    ("finance", "Finance"),
                    ("healthcare", "Healthcare"),
                    ("education", "Education"),
                    ("marketing", "Marketing"),
                    ("entertainment", "Entertainment"),
                    ("retail", "Retail"),
                    ("hospitality", "Hospitality"),
                    ("manufacturing", "Manufacturing"),
                    ("automotive", "Automotive"),
                    ("real_estate", "Real Estate"),
                    ("energy", "Energy"),
                    ("construction", "Construction"),
                    ("telecommunications", "Telecommunications"),
                    ("media", "Media"),
                    ("fashion", "Fashion"),
                    ("agriculture", "Agriculture"),
                    ("pharmaceutical", "Pharmaceutical"),
                    ("environment", "Environment"),
                    ("non_profit", "Non-Profit"),
                    ("government", "Government"),
                    ("consulting", "Consulting"),
                    ("transportation", "Transportation"),
                    ("sports", "Sports"),
                    ("food_beverage", "Food & Beverage"),
                    ("art_design", "Art & Design"),
                    ("law_legal", "Law & Legal"),
                    ("architecture", "Architecture"),
                    ("science", "Science"),
                    ("research", "Research"),
                    ("other", "Other"),
                ],
                max_length=20,
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="desired_job_description",
            field=models.TextField(default="Not specified"),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="desired_job_title",
            field=models.CharField(default="Not specified", max_length=100),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="desired_location",
            field=models.CharField(
                blank=True,
                choices=[
                    ("remote_anywhere", "Remote Anywhere"),
                    ("remote_local", "Remote (local)"),
                    ("hybrid", "Hybrid"),
                    ("In-person", "In-person"),
                ],
                max_length=30,
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="profilepreferences",
            name="urgency",
            field=models.CharField(
                blank=True,
                choices=[
                    ("very_urgent", "Very Urgent"),
                    ("somewhat_urgent", "Somewhat Urgent"),
                    ("not_urgent", "Not Urgent"),
                ],
                max_length=20,
                null=True,
            ),
        ),
    ]

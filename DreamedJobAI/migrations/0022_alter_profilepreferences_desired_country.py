# Generated by Django 4.2.2 on 2023-08-21 19:35

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("DreamedJobAI", "0021_alter_profilepreferences_desired_country"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profilepreferences",
            name="desired_country",
            field=models.CharField(
                choices=[
                    ("Aruba", "Aruba"),
                    ("Anguilla", "Anguilla"),
                    ("Antigua and Barbuda", "Antigua and Barbuda"),
                    (
                        "Bonaire, Sint Eustatius and Saba",
                        "Bonaire, Sint Eustatius and Saba",
                    ),
                    ("Bahamas", "Bahamas"),
                    ("Saint Barthélemy", "Saint Barthélemy"),
                    ("Belize", "Belize"),
                    ("Bermuda", "Bermuda"),
                    ("Barbados", "Barbados"),
                    ("Canada", "Canada"),
                    ("Costa Rica", "Costa Rica"),
                    ("Cuba", "Cuba"),
                    ("Curacao", "Curacao"),
                    ("Cayman Islands", "Cayman Islands"),
                    ("Dominica", "Dominica"),
                    ("Dominican Republic", "Dominican Republic"),
                    ("Guadeloupe", "Guadeloupe"),
                    ("Grenada", "Grenada"),
                    ("Greenland", "Greenland"),
                    ("Guatemala", "Guatemala"),
                    ("Honduras", "Honduras"),
                    ("Haiti", "Haiti"),
                    ("Jamaica", "Jamaica"),
                    ("Saint Kitts and Nevis", "Saint Kitts and Nevis"),
                    ("Saint Lucia", "Saint Lucia"),
                    ("Saint Martin (French part)", "Saint Martin (French part)"),
                    ("Mexico", "Mexico"),
                    ("Montserrat", "Montserrat"),
                    ("Martinique", "Martinique"),
                    ("Nicaragua", "Nicaragua"),
                    ("Panama", "Panama"),
                    ("Puerto Rico", "Puerto Rico"),
                    ("El Salvador", "El Salvador"),
                    ("Saint Pierre and Miquelon", "Saint Pierre and Miquelon"),
                    ("Turks and Caicos Islands", "Turks and Caicos Islands"),
                    ("Trinidad and Tobago", "Trinidad and Tobago"),
                    ("United States", "United States"),
                    (
                        "Saint Vincent and the Grenadines",
                        "Saint Vincent and the Grenadines",
                    ),
                    ("Virgin Islands, British", "Virgin Islands, British"),
                    ("Virgin Islands, U.S.", "Virgin Islands, U.S."),
                    ("Afghanistan", "Afghanistan"),
                    ("United Arab Emirates", "United Arab Emirates"),
                    ("Armenia", "Armenia"),
                    ("Azerbaijan", "Azerbaijan"),
                    ("Bangladesh", "Bangladesh"),
                    ("Bahrain", "Bahrain"),
                    ("Brunei Darussalam", "Brunei Darussalam"),
                    ("Bhutan", "Bhutan"),
                    ("Cocos (Keeling) Islands", "Cocos (Keeling) Islands"),
                    ("China", "China"),
                    ("Christmas Island", "Christmas Island"),
                    ("Cyprus", "Cyprus"),
                    ("Georgia", "Georgia"),
                    ("Hong Kong", "Hong Kong"),
                    ("Indonesia", "Indonesia"),
                    ("India", "India"),
                    (
                        "British Indian Ocean Territory",
                        "British Indian Ocean Territory",
                    ),
                    ("Iran, Islamic Republic of", "Iran, Islamic Republic of"),
                    ("Iraq", "Iraq"),
                    ("Israel", "Israel"),
                    ("Jordan", "Jordan"),
                    ("Japan", "Japan"),
                    ("Kazakhstan", "Kazakhstan"),
                    ("Kyrgyzstan", "Kyrgyzstan"),
                    ("Cambodia", "Cambodia"),
                    ("Korea", "Korea"),
                    ("Kuwait", "Kuwait"),
                    (
                        "Lao People's Democratic Republic",
                        "Lao People's Democratic Republic",
                    ),
                    ("Lebanon", "Lebanon"),
                    ("Sri Lanka", "Sri Lanka"),
                    ("Macao", "Macao"),
                    ("Maldives", "Maldives"),
                    ("Myanmar", "Myanmar"),
                    ("Mongolia", "Mongolia"),
                    ("Malaysia", "Malaysia"),
                    ("Nepal", "Nepal"),
                    ("Oman", "Oman"),
                    ("Pakistan", "Pakistan"),
                    ("Philippines", "Philippines"),
                    ("North Korea", "North Korea"),
                    ("Palestine, State of", "Palestine, State of"),
                    ("Qatar", "Qatar"),
                    ("Saudi Arabia", "Saudi Arabia"),
                    ("Singapore", "Singapore"),
                    ("Syrian Arab Republic", "Syrian Arab Republic"),
                    ("Thailand", "Thailand"),
                    ("Tajikistan", "Tajikistan"),
                    ("Turkmenistan", "Turkmenistan"),
                    ("Turkey", "Turkey"),
                    ("Taiwan", "Taiwan"),
                    ("Uzbekistan", "Uzbekistan"),
                    ("Vietnam", "Vietnam"),
                    ("Yemen", "Yemen"),
                    ("Angola", "Angola"),
                    ("Burundi", "Burundi"),
                    ("Benin", "Benin"),
                    ("Burkina Faso", "Burkina Faso"),
                    ("Botswana", "Botswana"),
                    ("Central African Republic", "Central African Republic"),
                    ("Côte d'Ivoire", "Côte d'Ivoire"),
                    ("Cameroon", "Cameroon"),
                    (
                        "Congo, The Democratic Republic of the",
                        "Congo, The Democratic Republic of the",
                    ),
                    ("Congo", "Congo"),
                    ("Comoros", "Comoros"),
                    ("Cabo Verde", "Cabo Verde"),
                    ("Djibouti", "Djibouti"),
                    ("Algeria", "Algeria"),
                    ("Egypt", "Egypt"),
                    ("Eritrea", "Eritrea"),
                    ("Ethiopia", "Ethiopia"),
                    ("Gabon", "Gabon"),
                    ("Ghana", "Ghana"),
                    ("Guinea", "Guinea"),
                    ("Gambia", "Gambia"),
                    ("Guinea-Bissau", "Guinea-Bissau"),
                    ("Equatorial Guinea", "Equatorial Guinea"),
                    ("Kenya", "Kenya"),
                    ("Liberia", "Liberia"),
                    ("Libya", "Libya"),
                    ("Lesotho", "Lesotho"),
                    ("Morocco", "Morocco"),
                    ("Madagascar", "Madagascar"),
                    ("Mali", "Mali"),
                    ("Mozambique", "Mozambique"),
                    ("Mauritania", "Mauritania"),
                    ("Mauritius", "Mauritius"),
                    ("Malawi", "Malawi"),
                    ("Mayotte", "Mayotte"),
                    ("Namibia", "Namibia"),
                    ("Niger", "Niger"),
                    ("Nigeria", "Nigeria"),
                    ("Reunion", "Reunion"),
                    ("Rwanda", "Rwanda"),
                    ("Sudan", "Sudan"),
                    ("Senegal", "Senegal"),
                    (
                        "Saint Helena, Ascension and Tristan da Cunha",
                        "Saint Helena, Ascension and Tristan da Cunha",
                    ),
                    ("Sierra Leone", "Sierra Leone"),
                    ("Somalia", "Somalia"),
                    ("South Sudan", "South Sudan"),
                    ("Sao Tome and Principe", "Sao Tome and Principe"),
                    ("Eswatini", "Eswatini"),
                    ("Seychelles", "Seychelles"),
                    ("Chad", "Chad"),
                    ("Togo", "Togo"),
                    ("Tunisia", "Tunisia"),
                    ("Tanzania", "Tanzania"),
                    ("Uganda", "Uganda"),
                    ("South Africa", "South Africa"),
                    ("Zambia", "Zambia"),
                    ("Zimbabwe", "Zimbabwe"),
                    ("Aland Islands", "Aland Islands"),
                    ("Albania", "Albania"),
                    ("Andorra", "Andorra"),
                    ("Austria", "Austria"),
                    ("Belgium", "Belgium"),
                    ("Bulgaria", "Bulgaria"),
                    ("Bosnia and Herzegovina", "Bosnia and Herzegovina"),
                    ("Belarus", "Belarus"),
                    ("Switzerland", "Switzerland"),
                    ("Czechia", "Czechia"),
                    ("Germany", "Germany"),
                    ("Denmark", "Denmark"),
                    ("Spain", "Spain"),
                    ("Estonia", "Estonia"),
                    ("Finland", "Finland"),
                    ("France", "France"),
                    ("Faroe Islands", "Faroe Islands"),
                    ("United Kingdom", "United Kingdom"),
                    ("Guernsey", "Guernsey"),
                    ("Gibraltar", "Gibraltar"),
                    ("Greece", "Greece"),
                    ("Croatia", "Croatia"),
                    ("Hungary", "Hungary"),
                    ("Isle of Man", "Isle of Man"),
                    ("Ireland", "Ireland"),
                    ("Iceland", "Iceland"),
                    ("Italy", "Italy"),
                    ("Jersey", "Jersey"),
                    ("Liechtenstein", "Liechtenstein"),
                    ("Lithuania", "Lithuania"),
                    ("Luxembourg", "Luxembourg"),
                    ("Latvia", "Latvia"),
                    ("Monaco", "Monaco"),
                    ("Moldova, Republic of", "Moldova, Republic of"),
                    ("North Macedonia", "North Macedonia"),
                    ("Malta", "Malta"),
                    ("Montenegro", "Montenegro"),
                    ("Netherlands", "Netherlands"),
                    ("Norway", "Norway"),
                    ("Poland", "Poland"),
                    ("Portugal", "Portugal"),
                    ("Romania", "Romania"),
                    ("Russian Federation", "Russian Federation"),
                    ("Svalbard and Jan Mayen", "Svalbard and Jan Mayen"),
                    ("San Marino", "San Marino"),
                    ("Serbia", "Serbia"),
                    ("Slovakia", "Slovakia"),
                    ("Slovenia", "Slovenia"),
                    ("Sweden", "Sweden"),
                    ("Ukraine", "Ukraine"),
                    ("Argentina", "Argentina"),
                    ("Bolivia", "Bolivia"),
                    ("Brazil", "Brazil"),
                    ("Chile", "Chile"),
                    ("Colombia", "Colombia"),
                    ("Ecuador", "Ecuador"),
                    ("Falkland Islands (Malvinas)", "Falkland Islands (Malvinas)"),
                    ("French Guiana", "French Guiana"),
                    ("Guyana", "Guyana"),
                    ("Peru", "Peru"),
                    ("Paraguay", "Paraguay"),
                    (
                        "South Georgia and the South Sandwich Islands",
                        "South Georgia and the South Sandwich Islands",
                    ),
                    ("Suriname", "Suriname"),
                    ("Uruguay", "Uruguay"),
                    ("Venezuela", "Venezuela"),
                    ("American Samoa", "American Samoa"),
                    ("Australia", "Australia"),
                    ("Cook Islands", "Cook Islands"),
                    ("Fiji", "Fiji"),
                    (
                        "Micronesia, Federated States of",
                        "Micronesia, Federated States of",
                    ),
                    ("Guam", "Guam"),
                    ("Kiribati", "Kiribati"),
                    ("Marshall Islands", "Marshall Islands"),
                    ("Northern Mariana Islands", "Northern Mariana Islands"),
                    ("New Caledonia", "New Caledonia"),
                    ("Norfolk Island", "Norfolk Island"),
                    ("Niue", "Niue"),
                    ("Nauru", "Nauru"),
                    ("New Zealand", "New Zealand"),
                    ("Palau", "Palau"),
                    ("Papua New Guinea", "Papua New Guinea"),
                    ("French Polynesia", "French Polynesia"),
                    ("Solomon Islands", "Solomon Islands"),
                    ("Tokelau", "Tokelau"),
                    ("Tonga", "Tonga"),
                    ("Tuvalu", "Tuvalu"),
                    ("Vanuatu", "Vanuatu"),
                    ("Wallis and Futuna", "Wallis and Futuna"),
                    ("Samoa", "Samoa"),
                    ("Bouvet Island", "Bouvet Island"),
                    (
                        "Heard Island and McDonald Islands",
                        "Heard Island and McDonald Islands",
                    ),
                ],
                max_length=50,
            ),
        ),
    ]

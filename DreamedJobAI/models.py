from django.db import models
from django.contrib.auth.models import User

# ----------------------------------------------/
# PDF MODEL---------------------------/
#-----------------------------------------------/


class UserCV(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    text = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to='pdf_files/', null=True)
    extracted_text = models.TextField(null=True)

    def __str__(self):
        return self.user.username
    
# ----------------------------------------------/
# Email is unique now---------------------------/
#-----------------------------------------------/

User._meta.get_field('email')._unique = True

# ----------------------------------------------/
# Profile --------------------------------------/
#-----------------------------------------------/

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    picture = models.ImageField(default='user_pp/user.png', upload_to='user_pp/', blank=True)
    contact_number = models.CharField(default='Not specified', blank=True)
    linkedin = models.CharField(default='Not specified', blank=True)
    github = models.CharField(default='Not specified', blank=True)
    website = models.CharField(default='Not specified', blank=True)
    other = models.CharField(default='Not specified', blank=True)

    def __str__(self):
        return self.user.username

# ----------------------------------------------/
# Preferences --------------------------------------/
#-----------------------------------------------/

COMPENSATION_CHOICES = (
    ('1-10000', '1-10,000'),
    ('10001-20000', '10,000-20,000'),
    ('20001-30000', '20,001-30,000'),
    ('30001-40000', '30,001-40,000'),
    ('40001-50000', '40,001-50,000'),
    ('50001-60000', '50,001-60,000'),
    ('60001-70000', '60,001-70,000'),
    ('70001-80000', '70,001-80,000'),
    ('80001-90000', '80,001-90,000'),
    ('90001-100000', '90,001-100,000'),
    ('100000+', '100,000+'),
)

LOCATION_CHOICES = (
    ('remote_anywhere', 'Remote Anywhere'),
    ('remote_local', 'Remote (local)'),
    ('hybrid', 'Hybrid'),
    ('In-person', 'In-person')
)

INDUSTRY_CHOICES = (
    ('technology', 'Technology'),
    ('finance', 'Finance'),
    ('healthcare', 'Healthcare'),
    ('education', 'Education'),
    ('marketing', 'Marketing'),
    ('entertainment', 'Entertainment'),
    ('retail', 'Retail'),
    ('hospitality', 'Hospitality'),
    ('manufacturing', 'Manufacturing'),
    ('automotive', 'Automotive'),
    ('real_estate', 'Real Estate'),
    ('energy', 'Energy'),
    ('construction', 'Construction'),
    ('telecommunications', 'Telecommunications'),
    ('media', 'Media'),
    ('fashion', 'Fashion'),
    ('agriculture', 'Agriculture'),
    ('pharmaceutical', 'Pharmaceutical'),
    ('environment', 'Environment'),
    ('non_profit', 'Non-Profit'),
    ('government', 'Government'),
    ('consulting', 'Consulting'),
    ('transportation', 'Transportation'),
    ('sports', 'Sports'),
    ('food_beverage', 'Food & Beverage'),
    ('art_design', 'Art & Design'),
    ('law_legal', 'Law & Legal'),
    ('architecture', 'Architecture'),
    ('science', 'Science'),
    ('research', 'Research'),
    ('other', 'Other'),
)

START_DAY_CHOICES = [
        ('Within 1 week', 'Within 1 week'),
        ('1-2 weeks', '1-2 weeks'),
        ('2-4 weeks', '2-4 weeks'),
        ('1-2 months', '1-2 months'),
        ('More than 2 months', 'More than 2 months'),
    ]

URGENCY_CHOICES = (
    ('very_urgent', 'Very Urgent'),
    ('somewhat_urgent', 'Somewhat Urgent'),
    ('not_urgent', 'Not Urgent'),
)

COUNTRY_CHOICES = (('Aruba', 'Aruba'), ('Anguilla', 'Anguilla'), ('Antigua and Barbuda', 'Antigua and Barbuda'), ('Bonaire, Sint Eustatius and Saba', 'Bonaire, Sint Eustatius and Saba'), ('Bahamas', 'Bahamas'), ('Saint Barthélemy', 'Saint Barthélemy'), ('Belize', 'Belize'), ('Bermuda', 'Bermuda'), ('Barbados', 'Barbados'), ('Canada', 'Canada'), ('Costa Rica', 'Costa Rica'), ('Cuba', 'Cuba'), ('Curacao', 'Curacao'), ('Cayman Islands', 'Cayman Islands'), ('Dominica', 'Dominica'), ('Dominican Republic', 'Dominican Republic'), ('Guadeloupe', 'Guadeloupe'), ('Grenada', 'Grenada'), ('Greenland', 'Greenland'), ('Guatemala', 'Guatemala'), ('Honduras', 'Honduras'), ('Haiti', 'Haiti'), ('Jamaica', 'Jamaica'), ('Saint Kitts and Nevis', 'Saint Kitts and Nevis'), ('Saint Lucia', 'Saint Lucia'), ('Saint Martin (French part)', 'Saint Martin (French part)'), ('Mexico', 'Mexico'), ('Montserrat', 'Montserrat'), ('Martinique', 'Martinique'), ('Nicaragua', 'Nicaragua'), ('Panama', 'Panama'), ('Puerto Rico', 'Puerto Rico'), ('El Salvador', 'El Salvador'), ('Saint Pierre and Miquelon', 'Saint Pierre and Miquelon'), ('Turks and Caicos Islands', 'Turks and Caicos Islands'), ('Trinidad and Tobago', 'Trinidad and Tobago'), ('United States', 'United States'), ('Saint Vincent and the Grenadines', 'Saint Vincent and the Grenadines'), ('Virgin Islands, British', 'Virgin Islands, British'), ('Virgin Islands, U.S.', 'Virgin Islands, U.S.'), ('Afghanistan', 'Afghanistan'), ('United Arab Emirates', 'United Arab Emirates'), ('Armenia', 'Armenia'), ('Azerbaijan', 'Azerbaijan'), ('Bangladesh', 'Bangladesh'), ('Bahrain', 'Bahrain'), ('Brunei Darussalam', 'Brunei Darussalam'), ('Bhutan', 'Bhutan'), ('Cocos (Keeling) Islands', 'Cocos (Keeling) Islands'), ('China', 'China'), ('Christmas Island', 'Christmas Island'), ('Cyprus', 'Cyprus'), ('Georgia', 'Georgia'), ('Hong Kong', 'Hong Kong'), ('Indonesia', 'Indonesia'), ('India', 'India'), ('British Indian Ocean Territory', 'British Indian Ocean Territory'), ('Iran, Islamic Republic of', 'Iran, Islamic Republic of'), ('Iraq', 'Iraq'), ('Israel', 'Israel'), ('Jordan', 'Jordan'), ('Japan', 'Japan'), ('Kazakhstan', 'Kazakhstan'), ('Kyrgyzstan', 'Kyrgyzstan'), ('Cambodia', 'Cambodia'), ('Korea', 'Korea'), ('Kuwait', 'Kuwait'), ("Lao People's Democratic Republic", "Lao People's Democratic Republic"), ('Lebanon', 'Lebanon'), ('Sri Lanka', 'Sri Lanka'), ('Macao', 'Macao'), ('Maldives', 'Maldives'), ('Myanmar', 'Myanmar'), ('Mongolia', 'Mongolia'), ('Malaysia', 'Malaysia'), ('Nepal', 'Nepal'), ('Oman', 'Oman'), ('Pakistan', 'Pakistan'), ('Philippines', 'Philippines'), ('North Korea', 'North Korea'), ('Palestine, State of', 'Palestine, State of'), ('Qatar', 'Qatar'), ('Saudi Arabia', 'Saudi Arabia'), ('Singapore', 'Singapore'), ('Syrian Arab Republic', 'Syrian Arab Republic'), ('Thailand', 'Thailand'), ('Tajikistan', 'Tajikistan'), ('Turkmenistan', 'Turkmenistan'), ('Turkey', 'Turkey'), ('Taiwan', 'Taiwan'), ('Uzbekistan', 'Uzbekistan'), ('Vietnam', 'Vietnam'), ('Yemen', 'Yemen'), ('Angola', 'Angola'), ('Burundi', 'Burundi'), ('Benin', 'Benin'), ('Burkina Faso', 'Burkina Faso'), ('Botswana', 'Botswana'), ('Central African Republic', 'Central African Republic'), ("Côte d'Ivoire", "Côte d'Ivoire"), ('Cameroon', 'Cameroon'), ('Congo, The Democratic Republic of the', 'Congo, The Democratic Republic of the'), ('Congo', 'Congo'), ('Comoros', 'Comoros'), ('Cabo Verde', 'Cabo Verde'), ('Djibouti', 'Djibouti'), ('Algeria', 'Algeria'), ('Egypt', 'Egypt'), ('Eritrea', 'Eritrea'), ('Ethiopia', 'Ethiopia'), ('Gabon', 'Gabon'), ('Ghana', 'Ghana'), ('Guinea', 'Guinea'), ('Gambia', 'Gambia'), ('Guinea-Bissau', 'Guinea-Bissau'), ('Equatorial Guinea', 'Equatorial Guinea'), ('Kenya', 'Kenya'), ('Liberia', 'Liberia'), ('Libya', 'Libya'), ('Lesotho', 'Lesotho'), ('Morocco', 'Morocco'), ('Madagascar', 'Madagascar'), ('Mali', 'Mali'), ('Mozambique', 'Mozambique'), ('Mauritania', 'Mauritania'), ('Mauritius', 'Mauritius'), ('Malawi', 'Malawi'), ('Mayotte', 'Mayotte'), ('Namibia', 'Namibia'), ('Niger', 'Niger'), ('Nigeria', 'Nigeria'), ('Reunion', 'Reunion'), ('Rwanda', 'Rwanda'), ('Sudan', 'Sudan'), ('Senegal', 'Senegal'), ('Saint Helena, Ascension and Tristan da Cunha', 'Saint Helena, Ascension and Tristan da Cunha'), ('Sierra Leone', 'Sierra Leone'), ('Somalia', 'Somalia'), ('South Sudan', 'South Sudan'), ('Sao Tome and Principe', 'Sao Tome and Principe'), ('Eswatini', 'Eswatini'), ('Seychelles', 'Seychelles'), ('Chad', 'Chad'), ('Togo', 'Togo'), ('Tunisia', 'Tunisia'), ('Tanzania', 'Tanzania'), ('Uganda', 'Uganda'), ('South Africa', 'South Africa'), ('Zambia', 'Zambia'), ('Zimbabwe', 'Zimbabwe'), ('Aland Islands', 'Aland Islands'), ('Albania', 'Albania'), ('Andorra', 'Andorra'), ('Austria', 'Austria'), ('Belgium', 'Belgium'), ('Bulgaria', 'Bulgaria'), ('Bosnia and Herzegovina', 'Bosnia and Herzegovina'), ('Belarus', 'Belarus'), ('Switzerland', 'Switzerland'), ('Czechia', 'Czechia'), ('Germany', 'Germany'), ('Denmark', 'Denmark'), ('Spain', 'Spain'), ('Estonia', 'Estonia'), ('Finland', 'Finland'), ('France', 'France'), ('Faroe Islands', 'Faroe Islands'), ('United Kingdom', 'United Kingdom'), ('Guernsey', 'Guernsey'), ('Gibraltar', 'Gibraltar'), ('Greece', 'Greece'), ('Croatia', 'Croatia'), ('Hungary', 'Hungary'), ('Isle of Man', 'Isle of Man'), ('Ireland', 'Ireland'), ('Iceland', 'Iceland'), ('Italy', 'Italy'), ('Jersey', 'Jersey'), ('Liechtenstein', 'Liechtenstein'), ('Lithuania', 'Lithuania'), ('Luxembourg', 'Luxembourg'), ('Latvia', 'Latvia'), ('Monaco', 'Monaco'), ('Moldova, Republic of', 'Moldova, Republic of'), ('North Macedonia', 'North Macedonia'), ('Malta', 'Malta'), ('Montenegro', 'Montenegro'), ('Netherlands', 'Netherlands'), ('Norway', 'Norway'), ('Poland', 'Poland'), ('Portugal', 'Portugal'), ('Romania', 'Romania'), ('Russian Federation', 'Russian Federation'), ('Svalbard and Jan Mayen', 'Svalbard and Jan Mayen'), ('San Marino', 'San Marino'), ('Serbia', 'Serbia'), ('Slovakia', 'Slovakia'), ('Slovenia', 'Slovenia'), ('Sweden', 'Sweden'), ('Ukraine', 'Ukraine'), ('Argentina', 'Argentina'), ('Bolivia', 'Bolivia'), ('Brazil', 'Brazil'), ('Chile', 'Chile'), ('Colombia', 'Colombia'), ('Ecuador', 'Ecuador'), ('Falkland Islands (Malvinas)', 'Falkland Islands (Malvinas)'), ('French Guiana', 'French Guiana'), ('Guyana', 'Guyana'), ('Peru', 'Peru'), ('Paraguay', 'Paraguay'), ('South Georgia and the South Sandwich Islands', 'South Georgia and the South Sandwich Islands'), ('Suriname', 'Suriname'), ('Uruguay', 'Uruguay'), ('Venezuela', 'Venezuela'), ('American Samoa', 'American Samoa'), ('Australia', 'Australia'), ('Cook Islands', 'Cook Islands'), ('Fiji', 'Fiji'), ('Micronesia, Federated States of', 'Micronesia, Federated States of'), ('Guam', 'Guam'), ('Kiribati', 'Kiribati'), ('Marshall Islands', 'Marshall Islands'), ('Northern Mariana Islands', 'Northern Mariana Islands'), ('New Caledonia', 'New Caledonia'), ('Norfolk Island', 'Norfolk Island'), ('Niue', 'Niue'), ('Nauru', 'Nauru'), ('New Zealand', 'New Zealand'), ('Palau', 'Palau'), ('Papua New Guinea', 'Papua New Guinea'), ('French Polynesia', 'French Polynesia'), ('Solomon Islands', 'Solomon Islands'), ('Tokelau', 'Tokelau'), ('Tonga', 'Tonga'), ('Tuvalu', 'Tuvalu'), ('Vanuatu', 'Vanuatu'), ('Wallis and Futuna', 'Wallis and Futuna'), ('Samoa', 'Samoa'), ('Bouvet Island', 'Bouvet Island'), ('Heard Island and McDonald Islands', 'Heard Island and McDonald Islands'))


class ProfilePreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    about = models.TextField(default="Not specified", blank=True)
    desired_job_title = models.CharField(default="Not specified", max_length=100, blank=True)
    desired_country = models.CharField(
        max_length=50,
        choices=COUNTRY_CHOICES,
        blank=False
    )
    desired_location = models.CharField(
        max_length=30,
        choices=LOCATION_CHOICES,
        blank=True,
        null=True
    )
    desired_job_description = models.TextField(default="Not specified", blank=True)
    desired_compensation = models.CharField(
        max_length=20,
        choices=COMPENSATION_CHOICES,
        blank=True,
        null=True
    )
    desired_benefits = models.CharField(default="Not specified", max_length=200, blank=True)
    desired_industry = models.CharField(
        max_length=20,
        choices=INDUSTRY_CHOICES,
        blank=True,
        null=True
    )
    desired_start_day = models.CharField(
        max_length=20,
        choices=START_DAY_CHOICES,
        blank=True,
        null=True
    )
    urgency = models.CharField(
        max_length=20,
        choices=URGENCY_CHOICES,
        blank=True,
        null=True
    )

    def __str__(self):
        return self.user.username
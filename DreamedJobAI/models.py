from django.db import models
from django.contrib.auth.models import User

# ----------------------------------------------/
# PDF MODEL---------------------------/
#-----------------------------------------------/


class UserCV(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    text = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to='pdf_files/', null=False)
    extracted_text = models.TextField(null=True)
    summary = models.TextField(null=True)

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


COUNTRY_CHOICES = (('Anywhere', 'Anywhere'), ('Afghanistan', 'Afghanistan'), ('Aland Islands', 'Aland Islands'), ('Albania', 'Albania'), ('Algeria', 'Algeria'), ('American Samoa', 'American Samoa'), ('Andorra', 'Andorra'), ('Angola', 'Angola'), ('Anguilla', 'Anguilla'), ('Antigua and Barbuda', 'Antigua and Barbuda'), ('Argentina', 'Argentina'), ('Armenia', 'Armenia'), ('Aruba', 'Aruba'), ('Australia', 'Australia'), ('Austria', 'Austria'), ('Azerbaijan', 'Azerbaijan'), ('Bahamas', 'Bahamas'), ('Bahrain', 'Bahrain'), ('Bangladesh', 'Bangladesh'), ('Barbados', 'Barbados'), ('Belarus', 'Belarus'), ('Belgium', 'Belgium'), ('Belize', 'Belize'), ('Benin', 'Benin'), ('Bermuda', 'Bermuda'), ('Bhutan', 'Bhutan'), ('Bolivia', 'Bolivia'), ('Bonaire, Sint Eustatius and Saba', 'Bonaire, Sint Eustatius and Saba'), ('Bosnia and Herzegovina', 'Bosnia and Herzegovina'), ('Botswana', 'Botswana'), ('Bouvet Island', 'Bouvet Island'), ('Brazil', 'Brazil'), ('British Indian Ocean Territory', 'British Indian Ocean Territory'), ('Brunei Darussalam', 'Brunei Darussalam'), ('Bulgaria', 'Bulgaria'), ('Burkina Faso', 'Burkina Faso'), ('Burundi', 'Burundi'), ('Cabo Verde', 'Cabo Verde'), ('Cambodia', 'Cambodia'), ('Cameroon', 'Cameroon'), ('Canada', 'Canada'), ('Cayman Islands', 'Cayman Islands'), ('Central African Republic', 'Central African Republic'), ('Chad', 'Chad'), ('Chile', 'Chile'), ('China', 'China'), ('Christmas Island', 'Christmas Island'), ('Cocos (Keeling) Islands', 'Cocos (Keeling) Islands'), ('Colombia', 'Colombia'), ('Comoros', 'Comoros'), ('Congo', 'Congo'), ('Congo, The Democratic Republic of the', 'Congo, The Democratic Republic of the'), ('Cook Islands', 'Cook Islands'), ('Costa Rica', 'Costa Rica'), ('Croatia', 'Croatia'), ('Cuba', 'Cuba'), ('Curacao', 'Curacao'), ('Cyprus', 'Cyprus'), ('Czechia', 'Czechia'), ("Côte d'Ivoire", "Côte d'Ivoire"), ('Denmark', 'Denmark'), ('Djibouti', 'Djibouti'), ('Dominica', 'Dominica'), ('Dominican Republic', 'Dominican Republic'), ('Ecuador', 'Ecuador'), ('Egypt', 'Egypt'), ('El Salvador', 'El Salvador'), ('Equatorial Guinea', 'Equatorial Guinea'), ('Eritrea', 'Eritrea'), ('Estonia', 'Estonia'), ('Eswatini', 'Eswatini'), ('Ethiopia', 'Ethiopia'), ('Falkland Islands (Malvinas)', 'Falkland Islands (Malvinas)'), ('Faroe Islands', 'Faroe Islands'), ('Fiji', 'Fiji'), ('Finland', 'Finland'), ('France', 'France'), ('French Guiana', 'French Guiana'), ('French Polynesia', 'French Polynesia'), ('Gabon', 'Gabon'), ('Gambia', 'Gambia'), ('Georgia', 'Georgia'), ('Germany', 'Germany'), ('Ghana', 'Ghana'), ('Gibraltar', 'Gibraltar'), ('Greece', 'Greece'), ('Greenland', 'Greenland'), ('Grenada', 'Grenada'), ('Guadeloupe', 'Guadeloupe'), ('Guam', 'Guam'), ('Guatemala', 'Guatemala'), ('Guernsey', 'Guernsey'), ('Guinea', 'Guinea'), ('Guinea-Bissau', 'Guinea-Bissau'), ('Guyana', 'Guyana'), ('Haiti', 'Haiti'), ('Heard Island and McDonald Islands', 'Heard Island and McDonald Islands'), ('Honduras', 'Honduras'), ('Hong Kong', 'Hong Kong'), ('Hungary', 'Hungary'), ('Iceland', 'Iceland'), ('India', 'India'), ('Indonesia', 'Indonesia'), ('Iran, Islamic Republic of', 'Iran, Islamic Republic of'), ('Iraq', 'Iraq'), ('Ireland', 'Ireland'), ('Isle of Man', 'Isle of Man'), ('Israel', 'Israel'), ('Italy', 'Italy'), ('Jamaica', 'Jamaica'), ('Japan', 'Japan'), ('Jersey', 'Jersey'), ('Jordan', 'Jordan'), ('Kazakhstan', 'Kazakhstan'), ('Kenya', 'Kenya'), ('Kiribati', 'Kiribati'), ('Korea', 'Korea'), ('Kuwait', 'Kuwait'), ('Kyrgyzstan', 'Kyrgyzstan'), ("Lao People's Democratic Republic", "Lao People's Democratic Republic"), ('Latvia', 'Latvia'), ('Lebanon', 'Lebanon'), ('Lesotho', 'Lesotho'), ('Liberia', 'Liberia'), ('Libya', 'Libya'), ('Liechtenstein', 'Liechtenstein'), ('Lithuania', 'Lithuania'), ('Luxembourg', 'Luxembourg'), ('Macao', 'Macao'), ('Madagascar', 'Madagascar'), ('Malawi', 'Malawi'), ('Malaysia', 'Malaysia'), ('Maldives', 'Maldives'), ('Mali', 'Mali'), ('Malta', 'Malta'), ('Marshall Islands', 'Marshall Islands'), ('Martinique', 'Martinique'), ('Mauritania', 'Mauritania'), ('Mauritius', 'Mauritius'), ('Mayotte', 'Mayotte'), ('Mexico', 'Mexico'), ('Micronesia, Federated States of', 'Micronesia, Federated States of'), ('Moldova, Republic of', 'Moldova, Republic of'), ('Monaco', 'Monaco'), ('Mongolia', 'Mongolia'), ('Montenegro', 'Montenegro'), ('Montserrat', 'Montserrat'), ('Morocco', 'Morocco'), ('Mozambique', 'Mozambique'), ('Myanmar', 'Myanmar'), ('Namibia', 'Namibia'), ('Nauru', 'Nauru'), ('Nepal', 'Nepal'), ('Netherlands', 'Netherlands'), ('New Caledonia', 'New Caledonia'), ('New Zealand', 'New Zealand'), ('Nicaragua', 'Nicaragua'), ('Niger', 'Niger'), ('Nigeria', 'Nigeria'), ('Niue', 'Niue'), ('Norfolk Island', 'Norfolk Island'), ('North Korea', 'North Korea'), ('North Macedonia', 'North Macedonia'), ('Northern Mariana Islands', 'Northern Mariana Islands'), ('Norway', 'Norway'), ('Oman', 'Oman'), ('Pakistan', 'Pakistan'), ('Palau', 'Palau'), ('Palestine, State of', 'Palestine, State of'), ('Panama', 'Panama'), ('Papua New Guinea', 'Papua New Guinea'), ('Paraguay', 'Paraguay'), ('Peru', 'Peru'), ('Philippines', 'Philippines'), ('Poland', 'Poland'), ('Portugal', 'Portugal'), ('Puerto Rico', 'Puerto Rico'), ('Qatar', 'Qatar'), ('Reunion', 'Reunion'), ('Romania', 'Romania'), ('Russian Federation', 'Russian Federation'), ('Rwanda', 'Rwanda'), ('Saint Barthélemy', 'Saint Barthélemy'), ('Saint Helena, Ascension and Tristan da Cunha', 'Saint Helena, Ascension and Tristan da Cunha'), ('Saint Kitts and Nevis', 'Saint Kitts and Nevis'), ('Saint Lucia', 'Saint Lucia'), ('Saint Martin (French part)', 'Saint Martin (French part)'), ('Saint Pierre and Miquelon', 'Saint Pierre and Miquelon'), ('Saint Vincent and the Grenadines', 'Saint Vincent and the Grenadines'), ('Samoa', 'Samoa'), ('San Marino', 'San Marino'), ('Sao Tome and Principe', 'Sao Tome and Principe'), ('Saudi Arabia', 'Saudi Arabia'), ('Senegal', 'Senegal'), ('Serbia', 'Serbia'), ('Seychelles', 'Seychelles'), ('Sierra Leone', 'Sierra Leone'), ('Singapore', 'Singapore'), ('Slovakia', 'Slovakia'), ('Slovenia', 'Slovenia'), ('Solomon Islands', 'Solomon Islands'), ('Somalia', 'Somalia'), ('South Africa', 'South Africa'), ('South Georgia and the South Sandwich Islands', 'South Georgia and the South Sandwich Islands'), ('South Sudan', 'South Sudan'), ('Spain', 'Spain'), ('Sri Lanka', 'Sri Lanka'), ('Sudan', 'Sudan'), ('Suriname', 'Suriname'), ('Svalbard and Jan Mayen', 'Svalbard and Jan Mayen'), ('Sweden', 'Sweden'), ('Switzerland', 'Switzerland'), ('Syrian Arab Republic', 'Syrian Arab Republic'), ('Taiwan', 'Taiwan'), ('Tajikistan', 'Tajikistan'), ('Tanzania', 'Tanzania'), ('Thailand', 'Thailand'), ('Togo', 'Togo'), ('Tokelau', 'Tokelau'), ('Tonga', 'Tonga'), ('Trinidad and Tobago', 'Trinidad and Tobago'), ('Tunisia', 'Tunisia'), ('Turkey', 'Turkey'), ('Turkmenistan', 'Turkmenistan'), ('Turks and Caicos Islands', 'Turks and Caicos Islands'), ('Tuvalu', 'Tuvalu'), ('Uganda', 'Uganda'), ('Ukraine', 'Ukraine'), ('United Arab Emirates', 'United Arab Emirates'), ('United Kingdom', 'United Kingdom'), ('United States', 'United States'), ('Uruguay', 'Uruguay'), ('Uzbekistan', 'Uzbekistan'), ('Vanuatu', 'Vanuatu'), ('Venezuela', 'Venezuela'), ('Vietnam', 'Vietnam'), ('Virgin Islands, British', 'Virgin Islands, British'), ('Virgin Islands, U.S.', 'Virgin Islands, U.S.'), ('Wallis and Futuna', 'Wallis and Futuna'), ('Yemen', 'Yemen'), ('Zambia', 'Zambia'), ('Zimbabwe', 'Zimbabwe'))

class UserProfilePreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    about = models.TextField(default="Not specified", blank=True)
    desired_job_title = models.CharField(default="Not specified", max_length=100, blank=True)
    
    desired_country = models.CharField(
        max_length=50,
        choices=COUNTRY_CHOICES,
        blank=False
    )
    second_desired_country = models.CharField(
        max_length=50,
        choices=COUNTRY_CHOICES,
        blank=True
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


# ----------------------------------------------/
# Job --------------------------------------/
#-----------------------------------------------/

class SuitableJobs(models.Model):
    job_id = models.IntegerField()
    title = models.CharField(max_length=200)
    link = models.URLField(max_length=1000)
    location = models.CharField(max_length=200)
    summary = models.TextField()
    user_id = models.IntegerField()
    suitability = models.CharField(max_length=200)
    explanation = models.TextField()
    pubdate = models.DateTimeField()

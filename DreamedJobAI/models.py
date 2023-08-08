from django.db import models
from django.contrib.auth.models import User

# ----------------------------------------------/
# PDF MODEL---------------------------/
#-----------------------------------------------/

class UserText(models.Model):
    text = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    pdf_file = models.FileField(upload_to='pdf_files/', null=True)
    extracted_text = models.TextField(null=True)

    def __str__(self):
        return self.text
    
# ----------------------------------------------/
# Email is unique now---------------------------/
#-----------------------------------------------/

User._meta.get_field('email')._unique = True

# ----------------------------------------------/
# Profile --------------------------------------/
#-----------------------------------------------/

COUNTRY_CHOICES = (
    ('afghanistan', 'Afghanistan'),
    ('albania', 'Albania'),
    ('algeria', 'Algeria'),
    ('andorra', 'Andorra'),
    ('angola', 'Angola'),
    ('antigua_barbuda', 'Antigua and Barbuda'),
    ('argentina', 'Argentina'),
    ('armenia', 'Armenia'),
    ('australia', 'Australia'),
    ('austria', 'Austria'),
    ('azerbaijan', 'Azerbaijan'),
    ('bahamas', 'Bahamas'),
    ('bahrain', 'Bahrain'),
    ('bangladesh', 'Bangladesh'),
    ('barbados', 'Barbados'),
    ('belarus', 'Belarus'),
    ('belgium', 'Belgium'),
    ('belize', 'Belize'),
    ('benin', 'Benin'),
    ('bhutan', 'Bhutan'),
    ('bolivia', 'Bolivia'),
    ('bosnia_herzegovina', 'Bosnia and Herzegovina'),
    ('botswana', 'Botswana'),
    ('brazil', 'Brazil'),
    ('brunei', 'Brunei'),
    ('bulgaria', 'Bulgaria'),
    ('burkina_faso', 'Burkina Faso'),
    ('burundi', 'Burundi'),
    ('cabo_verde', 'Cabo Verde'),
    ('cambodia', 'Cambodia'),
    ('cameroon', 'Cameroon'),
    ('canada', 'Canada'),
    ('central_african_republic', 'Central African Republic'),
    ('chad', 'Chad'),
    ('chile', 'Chile'),
    ('china', 'China'),
    ('colombia', 'Colombia'),
    ('comoros', 'Comoros'),
    ('congo', 'Congo'),
    ('costa_rica', 'Costa Rica'),
    ('croatia', 'Croatia'),
    ('cuba', 'Cuba'),
    ('cyprus', 'Cyprus'),
    ('czech_republic', 'Czech Republic'),
    ('denmark', 'Denmark'),
    ('djibouti', 'Djibouti'),
    ('dominica', 'Dominica'),
    ('dominican_republic', 'Dominican Republic'),
    ('ecuador', 'Ecuador'),
    ('egypt', 'Egypt'),
    ('el_salvador', 'El Salvador'),
    ('equatorial_guinea', 'Equatorial Guinea'),
    ('eritrea', 'Eritrea'),
    ('estonia', 'Estonia'),
    ('eswatini', 'Eswatini'),
    ('ethiopia', 'Ethiopia'),
    ('fiji', 'Fiji'),
    ('finland', 'Finland'),
    ('france', 'France'),
    ('gabon', 'Gabon'),
    ('gambia', 'Gambia'),
    ('georgia', 'Georgia'),
    ('germany', 'Germany'),
    ('ghana', 'Ghana'),
    ('greece', 'Greece'),
    ('grenada', 'Grenada'),
    ('guatemala', 'Guatemala'),
    ('guinea', 'Guinea'),
    ('guinea_bissau', 'Guinea-Bissau'),
    ('guyana', 'Guyana'),
    ('haiti', 'Haiti'),
    ('honduras', 'Honduras'),
    ('hungary', 'Hungary'),
    ('iceland', 'Iceland'),
    ('india', 'India'),
    ('indonesia', 'Indonesia'),
    ('iran', 'Iran'),
    ('iraq', 'Iraq'),
    ('ireland', 'Ireland'),
    ('israel', 'Israel'),
    ('italy', 'Italy'),
    ('ivory_coast', 'Ivory Coast'),
    ('jamaica', 'Jamaica'),
    ('japan', 'Japan'),
    ('jordan', 'Jordan'),
    ('kazakhstan', 'Kazakhstan'),
    ('kenya', 'Kenya'),
    ('kiribati', 'Kiribati'),
    ('north_korea', 'North Korea'),
    ('south_korea', 'South Korea'),
    ('kosovo', 'Kosovo'),
    ('kuwait', 'Kuwait'),
    ('kyrgyzstan', 'Kyrgyzstan'),
    ('laos', 'Laos'),
    ('latvia', 'Latvia'),
    ('lebanon', 'Lebanon'),
    ('lesotho', 'Lesotho'),
    ('liberia', 'Liberia'),
    ('libya', 'Libya'),
    ('liechtenstein', 'Liechtenstein'),
    ('lithuania', 'Lithuania'),
    ('luxembourg', 'Luxembourg'),
    ('madagascar', 'Madagascar'),
    ('malawi', 'Malawi'),
    ('malaysia', 'Malaysia'),
    ('maldives', 'Maldives'),
    ('mali', 'Mali'),
    ('malta', 'Malta'),
    ('marshall_islands', 'Marshall Islands'),
    ('mauritania', 'Mauritania'),
    ('mauritius', 'Mauritius'),
    ('mexico', 'Mexico'),
    ('micronesia', 'Micronesia'),
    ('moldova', 'Moldova'),
    ('monaco', 'Monaco'),
    ('mongolia', 'Mongolia'),
    ('montenegro', 'Montenegro'),
    ('morocco', 'Morocco'),
    ('mozambique', 'Mozambique'),
    ('myanmar', 'Myanmar'),
    ('namibia', 'Namibia'),
    ('nauru', 'Nauru'),
    ('nepal', 'Nepal'),
    ('netherlands', 'Netherlands'),
    ('new_zealand', 'New Zealand'),
    ('nicaragua', 'Nicaragua'),
    ('niger', 'Niger'),
    ('nigeria', 'Nigeria'),
    ('north_macedonia', 'North Macedonia'),
    ('norway', 'Norway'),
    ('oman', 'Oman'),
    ('pakistan', 'Pakistan'),
    ('palau', 'Palau'),
    ('panama', 'Panama'),
    ('papua_new_guinea', 'Papua New Guinea'),
    ('paraguay', 'Paraguay'),
    ('peru', 'Peru'),
    ('philippines', 'Philippines'),
    ('poland', 'Poland'),
    ('portugal', 'Portugal'),
    ('qatar', 'Qatar'),
    ('romania', 'Romania'),
    ('russia', 'Russia'),
    ('rwanda', 'Rwanda'),
    ('saint_kitts_nevis', 'Saint Kitts and Nevis'),
    ('saint_lucia', 'Saint Lucia'),
    ('saint_vincent_grenadines', 'Saint Vincent and the Grenadines'),
    ('samoa', 'Samoa'),
    ('san_marino', 'San Marino'),
    ('sao_tome_principe', 'Sao Tome and Principe'),
    ('saudi_arabia', 'Saudi Arabia'),
    ('senegal', 'Senegal'),
    ('serbia', 'Serbia'),
    ('seychelles', 'Seychelles'),
    ('sierra_leone', 'Sierra Leone'),
    ('singapore', 'Singapore'),
    ('slovakia', 'Slovakia'),
    ('slovenia', 'Slovenia'),
    ('solomon_islands', 'Solomon Islands'),
    ('somalia', 'Somalia'),
    ('south_africa', 'South Africa'),
    ('south_sudan', 'South Sudan'),
    ('spain', 'Spain'),
    ('sri_lanka', 'Sri Lanka'),
    ('sudan', 'Sudan'),
    ('suriname', 'Suriname'),
    ('sweden', 'Sweden'),
    ('switzerland', 'Switzerland'),
    ('syria', 'Syria'),
    ('taiwan', 'Taiwan'),
    ('tajikistan', 'Tajikistan'),
    ('tanzania', 'Tanzania'),
    ('thailand', 'Thailand'),
    ('timor_leste', 'Timor-Leste'),
    ('togo', 'Togo'),
    ('tonga', 'Tonga'),
    ('trinidad_tobago', 'Trinidad and Tobago'),
    ('tunisia', 'Tunisia'),
    ('turkey', 'Turkey'),
    ('turkmenistan', 'Turkmenistan'),
    ('tuvalu', 'Tuvalu'),
    ('uganda', 'Uganda'),
    ('ukraine', 'Ukraine'),
    ('united_arab_emirates', 'United Arab Emirates'),
    ('united_kingdom', 'United Kingdom'),
    ('united_states', 'United States'),
    ('uruguay', 'Uruguay'),
    ('uzbekistan', 'Uzbekistan'),
    ('vanuatu', 'Vanuatu'),
    ('vatican_city', 'Vatican City'),
    ('venezuela', 'Venezuela'),
    ('vietnam', 'Vietnam'),
    ('yemen', 'Yemen'),
    ('zambia', 'Zambia'),
    ('zimbabwe', 'Zimbabwe'),
)


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    picture = models.ImageField(default='user_pp/user.png', upload_to='user_pp/', blank=True)
    country = models.CharField(
        max_length=35,
        choices=COUNTRY_CHOICES,
        blank=True
    )
    state = models.CharField(default='Not specified', blank=True)
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


class ProfilePreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    about = models.TextField(default="Not specified", blank=True)
    desired_job_title = models.CharField(default="Not specified", max_length=100, blank=True)
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
        choices=URGENCY_CHOICES,
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
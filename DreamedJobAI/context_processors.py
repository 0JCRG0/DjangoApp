from .models import SuitableJobs

def jobs_processor(request):
    if request.user.is_authenticated:
        user_id = request.user.id
        jobs = SuitableJobs.objects.filter(user_id=user_id)
        has_jobs = jobs.exists()
    else:
        has_jobs = False

    return {'has_jobs': has_jobs}
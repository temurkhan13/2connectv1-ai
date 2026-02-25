from celery import Celery
from celery.schedules import crontab
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Override shell env vars with .env values

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

celery_app = Celery(
    'reciprocity_ai',
    broker=redis_url,
    backend=redis_url,
    include=[
        'app.workers.resume_processing', 
        'app.workers.persona_processing', 
        'app.workers.embedding_processing',
        'app.workers.scheduled_matching',
        'app.workers.ai_chat_processing'
    ]
)


celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Celery Beat Schedule (Periodic Tasks)
# No periodic scheduling; scheduled worker is triggered via API only

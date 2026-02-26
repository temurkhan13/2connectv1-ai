from celery import Celery
from celery.schedules import crontab
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # Override shell env vars with .env values

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# SSL configuration for rediss:// URLs (Upstash, etc.)
broker_ssl = None
backend_ssl = None
if redis_url.startswith('rediss://'):
    import ssl
    broker_ssl = {
        'ssl_cert_reqs': ssl.CERT_NONE
    }
    backend_ssl = broker_ssl

celery_app = Celery(
    'reciprocity_ai',
    broker=redis_url,
    backend=redis_url,
    broker_use_ssl=broker_ssl,
    redis_backend_use_ssl=backend_ssl,
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
    # Run tasks synchronously if no worker (staging/free tier)
    task_always_eager=os.getenv('CELERY_TASK_ALWAYS_EAGER', 'false').lower() == 'true',
    task_eager_propagates=os.getenv('CELERY_TASK_ALWAYS_EAGER', 'false').lower() == 'true',
)

# Celery Beat Schedule (Periodic Tasks)
# No periodic scheduling; scheduled worker is triggered via API only

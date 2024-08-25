from celery import shared_task
from .management.commands.import_articles import import_articles

@shared_task
def import_articles_task():
    import_articles()

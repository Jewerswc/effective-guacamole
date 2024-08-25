from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.utils.text import slugify

class RssFeed(models.Model):
    url = models.URLField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.url

class Articles(models.Model):
    title = models.CharField(max_length=255)
    link = models.CharField(max_length=255, unique=True)
    content = models.TextField()
    content_snippet = models.TextField(blank=True, null=True)
    interesting_score = models.FloatField(default=1)
    sw1_score = models.FloatField(default=1)
    marx_score = models.FloatField(default=1)
    hn_score = models.FloatField(default=1)
    sentiment = models.CharField(max_length=10, default='neutral')
    sentiment_value = models.FloatField(default=1)
    media_url = models.URLField(null=True, blank=True)
    published_at = models.DateTimeField(default=timezone.now)
    is_user_submitted = models.BooleanField(default=False)
    slug = models.SlugField(unique=True, blank=True, null=True)
    rss_feed = models.ForeignKey(RssFeed, on_delete=models.CASCADE, default=1)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

class UserArticleInteraction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    article = models.ForeignKey(Articles, on_delete=models.CASCADE)
    liked = models.BooleanField(default=False)
    disliked = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'article')


class Subscriber(models.Model):
    email = models.EmailField(unique=True)
    subscribed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email

class UserPreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    filters = models.JSONField(default=list)

    def __str__(self):
        return f'{self.user.username} Preferences'

User = get_user_model()

class NewsSource(models.Model):
    name = models.CharField(max_length=255)
    url = models.URLField()

    def __str__(self):
        return self.name


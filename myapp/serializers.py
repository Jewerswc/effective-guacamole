from rest_framework import serializers
from django.contrib.auth.models import User 
from .models import Articles, Subscriber, UserPreferences, NewsSource, RssFeed  # Import Subscriber model along with Articles

class NewsSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsSource
        fields = '__all__'

class RssFeedSerializer(serializers.ModelSerializer):
    class Meta:
        model = RssFeed
        fields = '__all__'

class ArticlesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Articles
        fields = '__all__'  # Ensures all fields are included

class SubscriberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subscriber
        fields = ['email']  # Only the email field is necessary

class UserPreferencesSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), required=True)

    class Meta:
        model = UserPreferences
        fields = ['user', 'filters']
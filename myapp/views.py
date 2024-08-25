from django.shortcuts import render, get_object_or_404
from django.utils.text import slugify
from django.utils import timezone
from rest_framework import status, viewsets, permissions, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from django.contrib.auth import authenticate, get_user_model
from django.core.mail import send_mail
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from .models import Articles, RssFeed, NewsSource, UserArticleInteraction, Subscriber, UserPreferences
from rest_framework.authtoken.models import Token
import pickle
import pandas as pd
import logging
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from .serializers import NewsSourceSerializer, RssFeedSerializer, SubscriberSerializer, ArticlesSerializer, UserPreferencesSerializer
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import JsonResponse
import stripe
from celery import shared_task
import feedparser
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timezone
from django.db import transaction
from django.core.mail import send_mail, BadHeaderError
import logging
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from .serializers import SubscriberSerializer

nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load models and vectorizers
with open('/Users/william/Downloads/vectoriser2403.pkl', 'rb') as vec_file, open('/Users/william/Downloads/model2403.pkl', 'rb') as model_file:
    vectorizer = pickle.load(vec_file)
    classification_model = pickle.load(model_file)

with open('/Users/william/Downloads/vectoriserSW1.pkl', 'rb') as sw1_vec_file, open('/Users/william/Downloads/modelSW1.pkl', 'rb') as sw1_model_file:
    sw1_vectorizer = pickle.load(sw1_vec_file)
    sw1_classification_model = pickle.load(sw1_model_file)

with open('/Users/william/Downloads/vectoriserMarx.pkl', 'rb') as Marx_vec_file, open('/Users/william/Downloads/modelMarx.pkl', 'rb') as Marx_model_file:
    Marx_vectorizer = pickle.load(Marx_vec_file)
    Marx_classification_model = pickle.load(Marx_model_file)

with open('/Users/william/Downloads/vectoriserHN.pkl', 'rb') as HN_vec_file, open('/Users/william/Downloads/modelHN.pkl', 'rb') as HN_model_file:
    HN_vectorizer = pickle.load(HN_vec_file)
    HN_classification_model = pickle.load(HN_model_file)

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
summary_model = BartForConditionalGeneration.from_pretrained(model_name)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

User = get_user_model()

def health_check(request):
    return HttpResponse('OK', status=200)

class ArticlesViewSet(viewsets.ModelViewSet):
    queryset = Articles.objects.all()
    serializer_class = ArticlesSerializer

class SignupAPIView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        password = request.data.get('password')
        email = request.data.get('email')
        if not username or not password or not email:
            return Response({"error": "Username, password and email required"}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({"error": "Username already exists"}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        token, created = Token.objects.get_or_create(user=user)
        return Response({"token": token.key}, status=status.HTTP_201_CREATED)

class LoginAPIView(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get("username")
        password = request.data.get("password")
        user = authenticate(username=username, password=password)
        if user:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({"token": token.key}, status=status.HTTP_200_OK)
        return Response({"error": "Invalid Credentials"}, status=status.HTTP_400_BAD_REQUEST)


# Get an instance of a logger
logger = logging.getLogger(__name__)

class SubscribeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        logger.info("Received subscription request with data: %s", request.data)
        serializer = SubscriberSerializer(data=request.data)
        if serializer.is_valid():
            subscriber = serializer.save()
            email = subscriber.email  # Assuming your Subscriber model has an email field
            try:
                # Send a confirmation email
                send_mail(
                    'Welcome to Our Newsletter!',
                    'Thank you for subscribing. You will now receive updates from us.',
                    'from@example.com',
                    [email],
                    fail_silently=False
                )
                logger.info("Confirmation email sent to: %s", email)
                return Response({"message": "Thank you for subscribing!"}, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.error("Failed to send confirmation email: %s", str(e))
                return Response({"message": "Subscription successful, but failed to send confirmation email."}, status=status.HTTP_201_CREATED)
        else:
            logger.warning("Invalid subscription data: %s", serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def check_and_create_model(user):
    try:
        likes_count = UserArticleInteraction.objects.filter(user=user, liked=True).count()
        dislikes_count = UserArticleInteraction.objects.filter(user=user, disliked=True).count()

        if likes_count >= 2 and dislikes_count >= 2:  # Set threshold as per your requirement
            create_model_and_vectorizer(user)
    except Exception as e:
        logger.error(f"Error in check_and_create_model: {e}\n{traceback.format_exc()}")

def create_model_and_vectorizer(user):
    try:
        interactions = UserArticleInteraction.objects.filter(user=user).select_related('article').values('article__content', 'liked')
        contents = [interaction['article__content'] for interaction in interactions]
        labels = [interaction['liked'] for interaction in interactions]

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(contents)
        model = LogisticRegression()
        model.fit(X, labels)

        user_model_dir = f'user_models/{user.id}'
        os.makedirs(user_model_dir, exist_ok=True)
        with open(f'{user_model_dir}/model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(f'{user_model_dir}/vectorizer.pkl', 'wb') as vec_file:
            pickle.dump(vectorizer, vec_file)
    except Exception as e:
        logger.error(f"Error in create_model_and_vectorizer: {e}\n{traceback.format_exc()}")

class LikeArticleAPIView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        try:
            user = request.user
            article_id = request.data.get('article_id')
            if not article_id:
                logger.error("Article ID is required")
                return Response({"error": "Article ID is required"}, status=status.HTTP_400_BAD_REQUEST)

            # Ensure the article exists
            article = Articles.objects.filter(id=article_id, rss_feed__user=user).first()
            if not article:
                logger.debug(f"Article with ID {article_id} does not exist, attempting to fetch and add...")
                article = fetch_and_add_article(article_id, user)

            if not article:
                logger.error(f"Article with ID {article_id} could not be fetched or created")
                return Response({"error": "Article could not be fetched or created"}, status=status.HTTP_404_NOT_FOUND)

            logger.debug(f"Creating interaction for user {user.username} and article {article.id}")
            interaction, created = UserArticleInteraction.objects.get_or_create(user=user, article=article)
            interaction.liked = True
            interaction.disliked = False
            interaction.save()

            return Response({"status": "article liked"}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in LikeArticleAPIView: {e}\n{traceback.format_exc()}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def fetch_and_add_article(article_id, user):
    logger.debug(f"Fetching article with ID {article_id} for user {user.username}")
    try:
        rss_feed = RssFeed.objects.filter(user=user).first()
        if not rss_feed:
            logger.error("No RSS feed found for the user.")
            return None

        logger.debug(f"Using RSS feed URL: {rss_feed.url}")

        # Fetching the feed
        feed_data = feedparser.parse(rss_feed.url)
        for entry in feed_data.entries:
            logger.debug(f"Checking entry with link: {entry.link}")

            # Check if the entry matches the given article_id (assumption based on link containing article_id)
            if str(article_id) in entry.link or entry.link.endswith(str(article_id)):
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc) if 'published_parsed' in entry else timezone.now()
                
                # Ensure article creation
                article, created = Articles.objects.get_or_create(
                    link=entry.link,
                    defaults={
                        'title': entry.title,
                        'content': entry.summary,
                        'published_at': published_at,
                        'rss_feed': rss_feed
                    }
                )
                if created:
                    logger.debug(f"Article created with ID {article.id}: {article}")
                else:
                    logger.debug(f"Article already exists with ID {article.id}: {article}")

                # Explicitly check if the article is saved and its ID matches
                if Articles.objects.filter(id=article.id).exists():
                    logger.debug(f"Verified article exists with ID {article.id}")
                    return article
                else:
                    logger.error(f"Article with ID {article.id} does not exist in the database after creation.")
                    return None

        logger.error(f"Article with ID {article_id} not found in RSS feed.")
        return None

    except Exception as e:
        logger.error(f"Error fetching article: {e}\n{traceback.format_exc()}")
        return None

    
class DislikeArticleAPIView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        try:
            user = request.user
            article_id = request.data.get('article_id')
            if not article_id:
                logger.error("Article ID is required")
                return Response({"error": "Article ID is required"}, status=status.HTTP_400_BAD_REQUEST)

            try:
                article = Articles.objects.get(id=article_id, rss_feed__user=user)
            except Articles.DoesNotExist:
                logger.debug(f"Article with ID {article_id} does not exist, fetching and adding...")
                article = fetch_and_add_article(article_id, user)

            logger.debug(f"Creating interaction for user {user.username} and article {article.id}")
            interaction, created = UserArticleInteraction.objects.get_or_create(user=user, article=article)
            interaction.liked = False
            interaction.disliked = True
            interaction.save()

            return Response({"status": "article disliked"}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in DislikeArticleAPIView: {e}\n{traceback.format_exc()}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RankUserArticlesAPIView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        user = request.user
        user_model_dir = f'user_models/{user.id}'
        user_model_path = f'{user_model_dir}/model.pkl'
        user_vectorizer_path = f'{user_model_dir}/vectorizer.pkl'

        if not os.path.exists(user_model_path) or not os.path.exists(user_vectorizer_path):
            return Response({"error": "Model not trained yet. Please like or dislike some articles to train the model."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            with open(user_model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(user_vectorizer_path, 'rb') as vec_file:
                vectorizer = pickle.load(vec_file)
        except Exception as e:
            error_message = f"Error loading user model or vectorizer: {e}\n{traceback.format_exc()}"
            logger.error(error_message)
            return Response({"error": error_message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        articles = Articles.objects.filter(rss_feed__user=user).values()
        df = pd.DataFrame(articles)
        
        if df.empty:
            return Response({"error": "No articles found."}, status=status.HTTP_404_NOT_FOUND)

        contents = df['content'].fillna('')

        try:
            X = vectorizer.transform(contents)
            df['predicted_score'] = model.predict_proba(X)[:, 1]
            ranked_articles = df.sort_values(by='predicted_score', ascending=False)
            ranked_articles_dict = ranked_articles.to_dict(orient='records')

            return Response(ranked_articles_dict, status=status.HTTP_200_OK)
        except Exception as e:
            error_message = f"Error during ranking: {e}\n{traceback.format_exc()}"
            logger.error(error_message)
            return Response({"error": error_message}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

logger = logging.getLogger(__name__)

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def user_preferences(request):
    try:
        logger.debug(f"Request method: {request.method}")
        logger.debug(f"User: {request.user}")
        preferences, created = UserPreferences.objects.get_or_create(user=request.user)
        if created:
            logger.info(f"Created new UserPreferences for user {request.user.username}")
    except Exception as e:
        logger.error(f"Error getting or creating UserPreferences: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    if request.method == 'GET':
        try:
            serializer = UserPreferencesSerializer(preferences)
            logger.info(f"Returning UserPreferences for user {request.user.username}: {serializer.data}")
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error serializing UserPreferences: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    elif request.method == 'POST':
        try:
            data = request.data.copy()
            data['user'] = request.user.id
            logger.info(f"Received data to update UserPreferences for user {request.user.username}: {data}")
            serializer = UserPreferencesSerializer(preferences, data=data)
            if serializer.is_valid():
                serializer.save()
                logger.info(f"Updated UserPreferences for user {request.user.username}")
                return Response(serializer.data)
            logger.error(f"UserPreferencesSerializer validation errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error saving UserPreferences: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def preprocess(text):
    return text.lower().strip()

def summarize_text(article_text):
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=100, min_length=80, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def calculate_sentiment(text):
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment is 'negative'
    else:
        sentiment is 'neutral'
    return sentiment, compound_score

def calculate_scores(processed_text):
    content_vector = vectorizer.transform([processed_text])
    probability = classification_model.predict_proba(content_vector)[0][1]

    sw1_content_vector = sw1_vectorizer.transform([processed_text])
    sw1_probability = sw1_classification_model.predict_proba(sw1_content_vector)[0][1]

    Marx_content_vector = Marx_vectorizer.transform([processed_text])
    Marx_probability = Marx_classification_model.predict_proba(Marx_content_vector)[0][1]

    HN_content_vector = HN_vectorizer.transform([processed_text])
    HN_probability = HN_classification_model.predict_proba(HN_content_vector)[0][1]

    word_count = len(processed_text.split())
    if word_count < 250:
        penalty_factor = 0.5
        probability *= penalty_factor
        sw1_probability *= penalty_factor
        Marx_probability *= penalty_factor
        HN_probability *= penalty_factor

    penalty_percentage = {
        'penalty_word1': 0.1,
        'penalty_word2': 0.1,
    }
    for word, penalty in penalty_percentage.items():
        if word in processed_text:
            probability *= (1 - penalty)
            sw1_probability *= (1 - penalty)
            Marx_probability *= (1 - penalty)
            HN_probability *= (1 - penalty)

    scores = {
        'interesting_score': max(0, min(1, probability)),
        'sw1_score': max(0, min(1, sw1_probability)),
        'marx_score': max(0, min(1, Marx_probability)),
        'hn_score': max(0, min(1, HN_probability))
    }
    
    return scores

def article_detail(request, slug):
    article = get_object_or_404(Articles, slug=slug, is_user_submitted=False)
    article_data = {
        'title': article.title,
        'content': article.content,
        'media_url': article.media_url,
        'published_at': article.published_at,
        'sentiment': article.sentiment,
        'sentiment_value': article.sentiment_value,
        'interesting_score': article.interesting_score,
        'sw1_score': article.sw1_score,
        'marx_score': article.marx_score,
        'hn_score': article.hn_score,
    }
    return JsonResponse(article_data)

class RssFeedDestroyAPIView(generics.DestroyAPIView):
    queryset = RssFeed.objects.all()
    serializer_class = RssFeedSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(user=self.request.user)

class NewsSourceListCreateAPIView(generics.ListCreateAPIView):
    queryset = NewsSource.objects.all()
    serializer_class = NewsSourceSerializer

class NewsSourceRetrieveUpdateDestroyAPIView(generics.RetrieveUpdateDestroyAPIView):
    queryset = NewsSource.objects.all()
    serializer_class = NewsSourceSerializer

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def follow_news_source(request):
    news_source_id = request.data.get('news_source_id')
    if not news_source_id:
        return Response({"error": "news_source_id is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    news_source = get_object_or_404(NewsSource, id=news_source_id)
    UserNewsSource.objects.get_or_create(user=request.user, news_source=news_source)
    return Response({"message": "News source followed successfully"}, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_rss_feed(request):
    rss_url = request.data.get('rss_url')
    if not rss_url:
        return Response({"error": "RSS URL is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Check if the RSS feed already exists
        existing_feed = RssFeed.objects.filter(url=rss_url, user=request.user).first()
        if existing_feed:
            return Response({"message": "RSS feed already exists"}, status=status.HTTP_200_OK)
        
        # If it doesn't exist, create a new feed and scrape
        rss_feed = RssFeed.objects.create(url=rss_url, user=request.user)
        return Response({"message": "RSS feed added successfully"}, status=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Error adding RSS feed: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RssFeedListCreateAPIView(generics.ListCreateAPIView):
    serializer_class = RssFeedSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return RssFeed.objects.filter(user=self.request.user)

class ArticlesListView(generics.ListAPIView):
    serializer_class = ArticlesSerializer

    def get_queryset(self):
        user = self.request.user
        rss_feeds = RssFeed.objects.filter(user=user)
        return Articles.objects.filter(rss_feed__in=rss_feeds)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_articles(request):
    user = request.user
    rss_feeds = RssFeed.objects.filter(user=user)
    articles = Articles.objects.filter(rss_feed__in=rss_feeds)
    serializer = ArticlesSerializer(articles, many=True)
    return Response(serializer.data)

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_article(request):
    if request.method == 'POST':
        title = request.data['title']
        content = request.data['content']
        image = request.FILES.get('image')

        processed_text = preprocess(content)
        scores = calculate_scores(processed_text)
        article_summary = summarize_text(content)
        sentiment, sentiment_value = calculate_sentiment(processed_text)

        # Save image if provided
        image_url = None
        if image:
            image_path = default_storage.save('images/' + image.name, ContentFile(image.read()))
            image_url = default_storage.url(image_path)

        slug = slugify(title)

        # Save the article to the database
        article = Articles.objects.create(
            title=title,
            content=content,
            content_snippet=article_summary,
            interesting_score=scores['interesting_score'],
            sw1_score=scores['sw1_score'],
            marx_score=scores['marx_score'],
            hn_score=scores['hn_score'],
            sentiment=sentiment,
            sentiment_value=sentiment_value,
            media_url=image_url,
            slug=slug,
            is_user_submitted=True
        )

        return Response({"message": "Article posted successfully", "slug": slug}, status=status.HTTP_200_OK)

    return Response({"error": "Invalid request"}, status=status.HTTP_400_BAD_REQUEST)

@shared_task
def import_articles_task():
    import logging
    from bs4 import BeautifulSoup
    import feedparser
    import requests
    from datetime import datetime, timezone
    from .models import RssFeed, Articles

    logger = logging.getLogger(__name__)

    logger.info('Starting import_articles_task')
    rss_feeds = RssFeed.objects.all()

    for feed in rss_feeds:
        logger.info(f'Processing feed: {feed.url}')
        url = feed.url
        user = feed.user

        feed_data = feedparser.parse(url)
        for entry in feed_data.entries:
            published_at = datetime.now(timezone.utc)  # Default value if 'published' is not available
            if 'published' in entry:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

            media_url = entry.media_content[0]['url'] if 'media_content' in entry and entry.media_content else None

            response = requests.get(entry.link)
            soup = BeautifulSoup(response.content, 'html.parser')
            article_text = ' '.join(p.get_text() for p in soup.find_all('p'))

            # Save the article
            article, created = Articles.objects.get_or_create(
                title=entry.title,
                link=entry.link,
                published_at=published_at,
                defaults={
                    'content': article_text,
                    'media_url': media_url,
                    'rss_feed': feed,  # Associate with the RSS feed
                }
            )
            if created:
                logger.info(f'Created new article: {entry.title}')
            else:
                logger.info(f'Article already exists: {entry.title}')

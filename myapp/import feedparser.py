import feedparser
from bs4 import BeautifulSoup
import requests
import psycopg2
from datetime import datetime, timezone
import boto3
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration
from django.core.management.base import BaseCommand
from myapp.models import RssFeed, Articles
from django.contrib.auth.models import User
from django.conf import settings

# S3 setup
s3 = boto3.client('s3')
bucket_name = 'interestingimportbucket'

def download_model_from_s3(key):
    s3.download_file(bucket_name, key, '/tmp/' + key)
    with open('/tmp/' + key, 'rb') as file:
        return pickle.load(file)

# Load vectorizer and model for classification
vectorizer = download_model_from_s3('vectoriser2403.pkl')
classification_model = download_model_from_s3('model2403.pkl')

# Load vectorizer and model for SW1 score
sw1_vectorizer = download_model_from_s3('vectoriserSW1.pkl')
sw1_classification_model = download_model_from_s3('modelSW1.pkl')

Marx_vectorizer = download_model_from_s3('vectoriserMarx.pkl')
Marx_classification_model = download_model_from_s3('modelMarx.pkl')

HN_vectorizer = download_model_from_s3('vectoriserHN.pkl')
HN_classification_model = download_model_from_s3('modelHN.pkl')

# Load pre-trained BART model and tokenizer for summarization
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
summary_model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess(text):
    return text.lower()

def summarize_text(article_text):
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summary_model.generate(inputs, max_length=100, min_length=80, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def parse_pubdate(pubdate_str):
    try:
        return datetime.strptime(pubdate_str, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"Date parsing error: {e}, using current time as fallback.")
        return datetime.now(timezone.utc)

class Command(BaseCommand):
    help = 'Imports articles from RSS feeds'

    def handle(self, *args, **kwargs):
        self.import_articles()

    def import_articles(self):
        conn = psycopg2.connect(
            dbname=settings.DATABASES['default']['NAME'],
            user=settings.DATABASES['default']['USER'],
            password=settings.DATABASES['default']['PASSWORD'],
            host=settings.DATABASES['default']['HOST'],
            port=settings.DATABASES['default']['PORT'],
            sslmode='require'
        )
        cursor = conn.cursor()

        for rss_feed in RssFeed.objects.all():
            feed = feedparser.parse(rss_feed.url)
            for entry in feed.entries:
                published_at = parse_pubdate(entry.published) if 'published' in entry else datetime.now(timezone.utc)
                media_url = entry.media_content[0]['url'] if 'media_content' in entry and entry.media_content else None

                response = requests.get(entry.link)
                soup = BeautifulSoup(response.content, 'html.parser')
                article_text = ' '.join(p.get_text() for p in soup.find_all('p'))
                processed_text = preprocess(article_text)

                # Calculate scores
                content_vector = vectorizer.transform([processed_text])
                probability = classification_model.predict_proba(content_vector)[0][1]

                sw1_content_vector = sw1_vectorizer.transform([processed_text])
                sw1_probability = sw1_classification_model.predict_proba(sw1_content_vector)[0][1]

                Marx_content_vector = Marx_vectorizer.transform([processed_text])
                Marx_probability = Marx_classification_model.predict_proba(Marx_content_vector)[0][1]

                HN_content_vector = HN_vectorizer.transform([processed_text])
                HN_probability = HN_classification_model.predict_proba(HN_content_vector)[0][1]

                # Penalize if the article text is under a certain word count
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

                # Ensure probabilities stay within 0 and 1
                probability = max(0, min(1, probability))
                sw1_probability = max(0, min(1, sw1_probability))
                Marx_probability = max(0, min(1, Marx_probability))
                HN_probability = max(0, min(1, HN_probability))

                # Generate a summary for the content_snippet field using BART
                article_summary = summarize_text(article_text)

                cursor.execute("""
                    INSERT INTO myapp_articles (title, link, published_at, media_url, content, content_snippet, interesting_score, sw1_score, marx_score, hn_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (link) DO NOTHING
                """, (entry.title, entry.link, published_at, media_url, article_text, article_summary, probability, sw1_probability, Marx_probability, HN_probability))
                conn.commit()

        cursor.close()
        conn.close()
        self.stdout.write(self.style.SUCCESS('Successfully imported articles'))


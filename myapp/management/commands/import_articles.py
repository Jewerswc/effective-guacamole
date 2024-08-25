from django.core.management.base import BaseCommand
from celery import shared_task
import feedparser
from bs4 import BeautifulSoup
import requests
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration
from datetime import datetime, timezone, timedelta
from myapp.models import Articles, RssFeed
from django.utils.text import slugify
import logging
import os

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Import articles from RSS feeds'

    def handle(self, *args, **kwargs):
        import_articles_task.delay()

@shared_task
def import_articles_task(feed_id=None):
    import_articles(feed_id)

def import_articles(feed_id=None):
    # Load vectorizer and model for classification from local files
    def load_local_model(filename):
        with open(os.path.join('/Users/william/Downloads/ebdjango/models', filename), 'rb') as file:
            return pickle.load(file)

    vectorizer = load_local_model('vectoriser2403.pkl')
    classification_model = load_local_model('model2403.pkl')

    # Load vectorizer and model for SW1 score
    sw1_vectorizer = load_local_model('vectoriserSW1.pkl')
    sw1_classification_model = load_local_model('modelSW1.pkl')

    Marx_vectorizer = load_local_model('vectoriserMarx.pkl')
    Marx_classification_model = load_local_model('modelMarx.pkl')

    HN_vectorizer = load_local_model('vectoriserHN.pkl')
    HN_classification_model = load_local_model('modelHN.pkl')

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
            logger.error(f"Date parsing error: {e}, using current time as fallback.")
            return datetime.now(timezone.utc)

    def generate_unique_slug(title):
        base_slug = slugify(title)
        slug = base_slug
        counter = 1
        while Articles.objects.filter(slug=slug).exists():
            slug = f"{base_slug}-{counter}"
            counter += 1
        return slug

    rss_feeds = RssFeed.objects.all()
    if feed_id:
        rss_feeds = rss_feeds.filter(id=feed_id)

    for feed in rss_feeds:
        url = feed.url
        user = feed.user

        # Check if feed was updated recently
        if feed.last_updated and feed.last_updated > datetime.now(timezone.utc) - timedelta(hours=1):
            logger.info(f"Feed {url} was updated recently, skipping.")
            continue

        feed_data = feedparser.parse(url)
        for entry in feed_data.entries:
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
                # Add more words and penalties as needed
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

            # Generate a unique slug
            slug = generate_unique_slug(entry.title)

            # Save the article
            Articles.objects.create(
                title=entry.title,
                link=entry.link,
                published_at=published_at,
                content=article_text,
                content_snippet=article_summary,
                interesting_score=probability,
                sw1_score=sw1_probability,
                marx_score=Marx_probability,
                hn_score=HN_probability,
                media_url=media_url,
                is_user_submitted=False,
                slug=slug,
                rss_feed=feed  # Associate with the RSS feed
            )

        # Update the last_updated field
        feed.last_updated = datetime.now(timezone.utc)
        feed.save()

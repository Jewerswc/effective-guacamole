import os
import django
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ebdjango.settings.py')  # Replace 'your_project_name' with the name of your Django project
django.setup()

from .models import UserArticleInteraction  # Use absolute import

def train_model():
    interactions = UserArticleInteraction.objects.select_related('article').all().values('article__content', 'liked')
    df = pd.DataFrame(interactions)

    if df.empty:
        print("Not enough data to train the model.")
        return

    df['label'] = df['liked'].apply(lambda x: 1 if x else 0)
    X = df['article__content']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X, y)

    with open('article_ranking_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("Model trained successfully.")

if __name__ == "__main__":
    train_model()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from myapp.models import Articles, UserArticleInteraction

# Load data from the database
interactions = UserArticleInteraction.objects.select_related('article').all().values('article__content', 'liked')
df = pd.DataFrame(interactions)

# Preprocess data
df['label'] = df['liked'].apply(lambda x: 1 if x else 0)
X = df['article__content']
y = df['label']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

def train_model():
    interactions = UserArticleInteraction.objects.select_related('article').all().values('article__content', 'liked')
    df = pd.DataFrame(interactions)

    if df.empty:
        return "Not enough data to train the model."

    df['label'] = df['liked'].apply(lambda x: 1 if x else 0)
    X = df['article__content']
    y = df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    model = LogisticRegression(class_weight='balanced')  # Add class_weight='balanced'
    model.fit(X, y)

    with open('article_ranking_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    return "Model trained successfully."

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(class_weight='balanced')  # Add class_weight='balanced'
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# Save the model and vectorizer
with open('article_ranking_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

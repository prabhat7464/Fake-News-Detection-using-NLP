import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Download NLTK resources
#nltk.download('stopwords')
#nltk.download('wordnet')

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Initialize WordNet Lemmatizer
lm = WordNetLemmatizer()

# Read the dataset
df = pd.read_csv("News_dataset.csv")
df = df.dropna()
df.reset_index(inplace=True)
df = df.drop(['id', 'text', 'author'], axis=1)

# Preprocess the text
corpus = []
for i in range(len(df)):
    review = re.sub(r'[^a-zA-Z0-9]', ' ', df['title'][i])  # Corrected regex
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in stop_words]  # Use stop_words instead of stopwords
    review = " ".join(review)
    corpus.append(review)

# TF-IDF Vectorization
tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()
y = df['label']

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, stratify=y)

# Train the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Make predictions
y_pred = rf.predict(x_test)

# Evaluate the model
accuracy_score_ = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy_score_)

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df = pd.read_csv("News_dataset.csv")
df = df.dropna()
df.reset_index(inplace=True)
df = df.drop(['id','text','author'],axis = 1)

sample_data = 'The quick brown fox jumps over the lazy dog'
sample_data = sample_data.split()
sample_data = [data.lower() for data in sample_data]
stopwords = stopwords.words('english')
sample_data = [data for data in sample_data if data not in stopwords]
ps = PorterStemmer()
sample_data_stemming = [ps.stem(data) for data in sample_data]
lm = WordNetLemmatizer()
sample_data_lemma = [lm.lemmatize(data) for data in sample_data]
tf = TfidfVectorizer()
lm = WordNetLemmatizer()
corpus = []
for i in range(len(df)):
    review = re.sub('^a-zA-Z0-9',' ', df['title'][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(x) for x in review if x not in stopwords]
    review = " ".join(review)
    corpus.append(review)


x = tf.fit_transform(corpus).toarray()
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, stratify=y)

rf=joblib.load('model.pkl')

class Evaluation:
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train_evaluation(self):
        y_pred_train = self.model.predict(self.x_train)
        acc_scr_train = accuracy_score(self.y_train, y_pred_train)
        con_mat_train = confusion_matrix(self.y_train, y_pred_train)
        class_rep_train = classification_report(self.y_train, y_pred_train)
        return acc_scr_train, con_mat_train, class_rep_train

    def test_evaluation(self):
        y_pred_test = self.model.predict(self.x_test)
        acc_scr_test = accuracy_score(self.y_test, y_pred_test)
        con_mat_test = confusion_matrix(self.y_test, y_pred_test)
        class_rep_test = classification_report(self.y_test, y_pred_test)
        return acc_scr_test, con_mat_test, class_rep_test

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def text_preprocessing_user(self):
        lm = WordNetLemmatizer()
        pred_data = [self.data]
        preprocess_data = []
        for data in pred_data:
            review = re.sub('^a-zA-Z0-9', ' ', data)
            review = review.lower()
            review = review.split()
            review = [lm.lemmatize(x) for x in review if x not in stopwords]
            review = " ".join(review)
            preprocess_data.append(review)
        return preprocess_data

class Prediction:
    def __init__(self, pred_data, model):
        self.pred_data = pred_data
        self.model = model

    def prediction_model(self):
        preprocess_data = Preprocessing(self.pred_data).text_preprocessing_user()
        data = tf.transform(preprocess_data)
        prediction = self.model.predict(data)
        if prediction[0] == 0:
            return "The News Is Fake"
        else:
            return "The News Is Real"

# Streamlit UI
st.title("News Authenticity Checker")

# User input
user_input = st.text_input("Enter a news title:", "FLYNN: Hillary Clinton, Big Woman on Campus - Breitbart")

# Prediction
if st.button("Check"):
    prediction = Prediction(user_input, rf).prediction_model()
    st.title("Prediction:", prediction)

# Model evaluation
if st.checkbox("Show Model Evaluation"):
    evaluator = Evaluation(rf, x_train, x_test, y_train, y_test)
    train_acc, train_conf_mat, train_class_rep = evaluator.train_evaluation()
    test_acc, test_conf_mat, test_class_rep = evaluator.test_evaluation()

    st.subheader("Model Evaluation on Training Data:")
    st.write("Accuracy Score:", train_acc)
    st.write("Confusion Matrix:", train_conf_mat)
    st.write("Classification Report:", train_class_rep)

    st.subheader("Model Evaluation on Testing Data:")
    st.write("Accuracy Score:", test_acc)
    st.write("Confusion Matrix:", test_conf_mat)
    st.write("Classification Report:", test_class_rep)
 
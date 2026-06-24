import pandas as pd
import re
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Download nltk data
nltk.download('stopwords')
nltk.download('wordnet')

print("Loading dataset...")

# Load dataset
df = pd.read_csv("dataset/Resume.csv")

# Keep only needed columns
df = df[['Resume_str', 'Category']]

print(df.head())

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+', '', text)

    # remove html tags
    text = re.sub(r'<.*?>', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

print("Cleaning resumes...")

# Apply cleaning
df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

# Features and labels
X = df['cleaned_resume']
y = df['Category']

print("Applying TF-IDF...")

# Better TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

# Better model for NLP
model = LinearSVC()

model.fit(X_train, y_train)

print("Training model...")

# Better model for NLP
model = LinearSVC()

model.fit(X_train, y_train)

# SAVE MODEL
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# SAVE VECTORIZER
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")

print("Making predictions...")

# Prediction
y_pred = model.predict(X_test)

print("Making predictions...")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))    
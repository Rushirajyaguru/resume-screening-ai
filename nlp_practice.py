from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "Python developer with machine learning skills"

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print(filtered_words)


stemmer = PorterStemmer()

words = ["playing", "developer", "running"]

for word in words:
    print(stemmer.stem(word))
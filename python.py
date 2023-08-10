# PREPROCESS TRAINING DATASET 

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# download NLTK data 

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# load JSON data 

with open('data.json', 'r', encoding='utf-8') as f:
    raw_data = f.read()

# preprocess JSON data 

def preprocess(data):
    # tokenize
    tokens = nltk.word_tokenize(data)

    # convert to lowercase
    tokens = [word.lower() for word in tokens]

    # remove stopwords, punctuation 
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    # Lemmatize words 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

processed_data = [preprocess(qa) for qa in raw_data.split('\n')]


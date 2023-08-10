# PREPROCESS TRAINING DATASET 
import numpy 
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

# TRAIN A ML MODEL 
import tensorflow 
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Set parameters 
vocab_size = 5000
embedding_dim = 64
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = len(processed_data)

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(processed_data)
word_index = tokenizer.word_index

# Create sequences
sequences = tokenizer.texts_to_sequences(processed_data)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Create training data
training_data = padded_sequences[:training_size]
training_labels = padded_sequences[:training_size]

# Build the model 
model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tensorflow.keras.layers.Dropout(0.2),
    tensorflow.keras.layers.Conv1D(64, 5, activation='relu'),
    tensorflow.keras.layers.MaxPooling1D(pool_size=4),
    tensorflow.keras.layers.LSTM(64),
    tensorflow.keras.layers.Dense(64, activation='relu'),
    tensorflow.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
num_epochs = 50
history = model.fit(training_data, training_labels, epochs=num_epochs, verbose=2)

# BUILD SIMPLE COMMAND LINE INTERFACE 

# Function that predicts answer 
def predict_answer(model, tokenizer, question):
    # Preprocess question
    question = preprocess(question)
    # Convert question to sequence
    sequence = tokenizer.texts_to_sequences([question])
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    # Predict answer
    pred = model.predict(padded_sequence)[0]
    # Get index of highest probability
    idx = numpy.argmax(pred)
    # Get answer
    answer = tokenizer.index_word[idx]
    return answer

# Start chatbot
while True:
    question = input('You: ')
    answer = predict_answer(model, tokenizer, question)
    print('SithBot:', answer)
import re
import os
import nltk

# Ensure nltk points to the correct data folder
nltk.data.path.append(os.getenv('NLTK_DATA', './nltk_data'))

# Example of using nltk resources (e.g., stopwords)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Clean and preprocess text
    text = re.sub(r'\W', ' ', text).lower()
    words = word_tokenize(text)
    return ' '.join(word for word in words if word not in stop_words)

def preprocess_data(df):
    # Apply text cleaning to 'body' column
    df['cleaned_body'] = df['body'].apply(clean_text)
    return df

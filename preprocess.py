import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

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

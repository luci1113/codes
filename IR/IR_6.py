#pip install nltk


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join the processed words back into a string
    processed_text = ' '.join(stemmed_words)

    return processed_text

# Example usage
text = "Text preprocessing is an important step in natural language processing."
processed_text = preprocess_text(text)
print(processed_text)

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

# Download necessary NLTK resources
nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess(file_content):
    """
    Loads and preprocesses the text file content.

    Args:
        file_content (bytes): The content of the text file.

    Returns:
        list: A list of sentences extracted from the text file.
    """
    try:
        text = file_content.decode('utf-8')
        sentences = sent_tokenize(text)
        logging.info(f"Preprocessed {len(sentences)} sentences from the text file.")
        return sentences
    except UnicodeDecodeError:
        logging.error("Error decoding the text file. Please ensure it's in a valid format.")
        raise

def extract_relevant_sentences(sentences, case_text, top_n=5):
    """
    Extracts the top N most relevant sentences from the given text based on similarity to the case text.

    Args:
        sentences (list): A list of sentences from the text.
        case_text (str): The case text to compare against.
        top_n (int, optional): The number of top relevant sentences to return. Defaults to 5.

    Returns:
        list: A list of tuples containing the top N relevant sentences and their similarity scores.
    """
    vectorizer = TfidfVectorizer()
    case_vector = vectorizer.fit_transform([case_text])
    sentence_vectors = vectorizer.transform(sentences)

    similarity_scores = sentence_vectors.dot(case_vector.T).toarray().ravel()
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    top_sentences = [(sentences[i], similarity_scores[i]) for i in top_indices]
    logging.info(f"Extracted {len(top_sentences)} relevant sentences for the case text.")
    return top_sentences

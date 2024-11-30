import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import sent_tokenize
import numpy as np


# Download NLTK data
nltk.download('punkt')

def preprocess_text(text):
    """Tokenize the text into sentences."""
    
    return sent_tokenize(text)
    

def compute_sentence_scores(sentences):
    """Compute sentence scores using TF-IDF."""
    # Convert sentences into a Bag-of-Words representation
   
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Apply TF-IDF to the Bag-of-Words representation
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(vectors)

    # Compute sentence scores by summing TF-IDF values
    scores = tfidf_matrix.sum(axis=1).A1  # Convert to a flat array
   
    return scores

def generate_summary(sentences, scores, num_sentences):
    """Generate a summary using the top N scored sentences."""
    # Get indices of top N sentences
   
    top_indices = np.argsort(scores)[-num_sentences:][::-1]  # Sort and reverse
    summary = [sentences[i] for i in top_indices]
    
    return " ".join(summary)

def summarize_text(text, num_sentences=3):
    """Summarize the input text."""
   
    sentences = preprocess_text(text)
    scores = compute_sentence_scores(sentences)
    
    return generate_summary(sentences, scores, num_sentences)


# Example input text
sample_text = """
Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction 
between computers and humans through natural language. The ultimate objective of NLP is to read, 
decipher, understand, and make sense of human language in a valuable way. Applications of NLP include 
chatbots, sentiment analysis, machine translation, and text summarization. NLP techniques are also used 
to extract useful insights from unstructured text data, making it a key component in modern AI systems.
"""

if __name__ == "__main__":
    print("Original Text:")
    print(sample_text)
    print("\nSummarized Text:")
    print(summarize_text(sample_text, num_sentences=3))

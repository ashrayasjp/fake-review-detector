from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src.config import MAX_FEATURES, VECTORIZER_PATH
from utils.logger import logger

def extract_features(corpus):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(corpus)
    
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"TF-IDF Vectorizer saved at {VECTORIZER_PATH}")
    
    return X

from utils.helpers import load_model
from src.data_preprocessing import clean_text
from src.config import MODEL_PATH, VECTORIZER_PATH
import pickle

def predict_review(text):
    model = load_model(MODEL_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    
    return model.predict(features)[0]

if __name__ == "__main__":
    sample = "This product was amazing, I got it for free!"
    print("Prediction:", predict_review(sample))

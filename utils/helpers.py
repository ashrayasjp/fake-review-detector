import pickle
import pandas as pd
from src.data_preprocessing import clean_text
from utils.logger import logger
from src.config import DATA_PATH

def load_dataset(path=DATA_PATH):
    """
    Load dataset, map CG/OR to fake/genuine, and clean text
    Works with large datasets without editing CSV
    """
    df = pd.read_csv(path)
    
    # Map labels
    df['label'] = df['label'].map({'OR': 'genuine', 'CG': 'fake'})
    
    # Use 'text_' column instead of 'review_text'
    df["cleaned"] = df["text_"].apply(clean_text)
    
    logger.info(f"Dataset loaded with {len(df)} reviews")
    return df

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved at {path}")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

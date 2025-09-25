from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.helpers import save_model, load_dataset
from src.feature_extraction import extract_features
from src.config import MODEL_PATH, TEST_SIZE, RANDOM_STATE
from utils.logger import logger

def train():
    logger.info("Loading real dataset")
    df = load_dataset()
    
    logger.info("Extracting TF-IDF features")
    X = extract_features(df["cleaned"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info("Training Logistic Regression model")
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    train()

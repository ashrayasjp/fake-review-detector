# Fake Review Detector 📝

A simple Python tool to detect fake or genuine product reviews using Machine Learning (Logistic Regression + TF-IDF) and a Streamlit interface.
  
The dataset used in this project is the Fake Reviews Dataset by mexwell
Source: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

---

1. **Clone the repository:**

```bash
git clone https://github.com/ashrayasjp/fake-review-detector.git
cd fakereviewdetector
```
2. **Create and activate a virtual environment (optional):**


```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create Files:**
   ```bash
   Ensure to create an empty run.log file inside the logs/ directory to store logs.
   ```
5. **Train the model**
   ```bash
   python -m src.train_model
   ```
6. **Run predictions (optional)**
   ```bash
   python -m src.predict
   ```
7. **Run the Streamlit app**
   ```bash
    streamlit run app/app.py
   ```
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string

# Optional: Use custom nltk data path
nltk.data.path.append("C:/Users/anind/OneDrive/nltk_clean_data")
nltk.download('stopwords', download_dir="C:/Users/anind/OneDrive/nltk_clean_data")

# Initialize components
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App title
st.title("📩 Email/SMS Spam Classifier")

# Input box
input_sms = st.text_input('Enter the message')

# Transform function
def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Button to trigger prediction
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # Preprocess and predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header("🚫 SPAM")
        else:
            st.header("✅ Not Spam")

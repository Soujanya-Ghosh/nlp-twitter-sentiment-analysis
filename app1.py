import streamlit as st
import pickle
import re

# Load saved files
model = pickle.load(open('sentiment_model.pkl', 'rb'))

tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

encoder = pickle.load(open('encoder.pkl', 'rb'))


# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Prediction function
def predict_sentiment(tweet):

    cleaned = clean_text(tweet)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    sentiment = encoder.inverse_transform(prediction)

    return sentiment[0]


# Streamlit UI
st.title("🧠 Twitter Sentiment Analysis App")

st.write("Enter a tweet and predict its sentiment")


tweet = st.text_area("Enter Tweet")


if st.button("Predict"):

    if tweet.strip() != "":

        result = predict_sentiment(tweet)

        if result == "Positive":

            st.success("😊 Positive Sentiment")

        elif result == "Negative":

            st.error("😠 Negative Sentiment")

    else:

        st.warning("Please enter a tweet")


st.sidebar.header("About")

st.sidebar.write(
    "This app predicts sentiment using TF-IDF and SVM."
)
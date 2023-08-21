import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = text.lower()
    return text


def main():
    data_path = "Twitter_Data.csv"  # Update with your data file path
    data = pd.read_csv(data_path)

    # Preprocess data
    data = data.dropna(subset=["clean_text"])
    data["clean_text"] = data["clean_text"].apply(preprocess_text)
    data = data.dropna(subset=["category"])

    X = data["clean_text"]
    y = data["category"]

    # Vectorize the text data
    vectorizer = CountVectorizer(lowercase=True, stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    # Train the classifier
    classifier = MultinomialNB()
    classifier.fit(X_vec, y)

    # Take user input
    user_input = input("Enter a text for sentiment analysis: ")
    user_input_clean = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input_clean])

    # Get prediction
    prediction = classifier.predict(user_input_vec)
    if prediction == -1:
        sentiment = "Negative"
    elif prediction == 0:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    print("Predicted Sentiment:", sentiment)


if __name__ == "__main__":
    main()

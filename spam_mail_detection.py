import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = {
    "message": [
        "Win money now",
        "Hello how are you",
        "Claim your free prize",
        "Meeting at 10 am",
        "Congratulations you won",
        "Let's have lunch tomorrow"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"].map({"spam": 1, "ham": 0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
msg = input("Enter an email message: ")
msg_vector = vectorizer.transform([msg])
result = model.predict(msg_vector)

if result[0] == 1:
    print("Prediction: Spam ❌")
else:
    print("Prediction: Not Spam ✅")


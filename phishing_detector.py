import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("emails.csv")

# Feature Extraction Function
def preprocess_email(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' URL ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

data["text"] = data["text"].apply(preprocess_email)

# Features and Labels
X = data["text"]
y = data["label"]

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.show()

# Test Custom Email
sample_email = input("\nEnter Email Text: ")
sample_email = preprocess_email(sample_email)

sample_vector = vectorizer.transform([sample_email])
prediction = model.predict(sample_vector)

if prediction[0] == "phishing":
    print("⚠️ This Email is PHISHING")
else:
    print("✅ This Email is SAFE")

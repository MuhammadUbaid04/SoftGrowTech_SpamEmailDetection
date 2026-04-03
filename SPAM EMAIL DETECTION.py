import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# SAMPLE DATA
# -----------------------------
df = pd.read_csv("emails.csv")
# -----------------------------
# 1. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# 2. LABEL ENCODING
# -----------------------------
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# -----------------------------
# 3. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# -----------------------------
# 4. TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 6. EVALUATE
# -----------------------------
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# -----------------------------
# 7. USER INPUT PREDICTION
# -----------------------------
while True:
    email = input("\nEnter an email (or type 'exit' to quit): ")

    if email.lower() == "exit":
        break

    email_clean = clean_text(email)
    email_vec = vectorizer.transform([email_clean])

    prediction = model.predict(email_vec)[0]

    if prediction == 1:
        print("SPAM")
    else:
        print("NOT SPAM")

import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fake reviews dataset.csv")

print(df.columns)

# Correct columns for this dataset
TEXT_COLUMN = "text_"
LABEL_COLUMN = "label"

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

# Convert labels
# OR = Original Review (Genuine)
# CG = Computer Generated (Fake)

df[LABEL_COLUMN] = df[LABEL_COLUMN].map({
    "OR": 0,
    "CG": 1
})

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")
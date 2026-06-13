import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("=" * 50)
print("Customer Review Analyzer")
print("=" * 50)

while True:

    review = input("\nEnter Review (type 'exit' to quit): ")

    if review.lower() == "exit":
        break

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    confidence = max(
        model.predict_proba(review_vector)[0]
    ) * 100

    if prediction == 1:
        print(f" Negative Review ({confidence:.2f}% confidence)")
    else:
        print(f" Positive Review ({confidence:.2f}% confidence)")
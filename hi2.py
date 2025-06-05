import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data("/Users/amrita/Desktop/hunarintern/spam.csv")

def preprocess_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model
def train_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model
def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
def predict_message(model, vectorizer, messages):
    tfidf = vectorizer.transform(messages)
    predictions = model.predict(tfidf)
    return ["Spam" if p == 1 else "Ham" for p in predictions]
if __name__ == "__main__":
    df = load_data("spam.csv")
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_data(df)
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    evaluate_model(nb_model, X_test_tfidf, y_test, name="Naive Bayes")
    svm_model = train_svm(X_train_tfidf, y_train)
    evaluate_model(svm_model, X_test_tfidf, y_test, name="SVM")
    sample_messages = [
        "Congratulations! You've won the lottery. Call now!",
        "Are we still on for the meeting tomorrow?"
    ]
    print("\nCustom Predictions:")
    for msg in sample_messages:
        nb_result = predict_message(nb_model, vectorizer, [msg])[0]
        svm_result = predict_message(svm_model, vectorizer, [msg])[0]
        print(f"Message: {msg}\n - Naive Bayes: {nb_result}\n - SVM: {svm_result}\n")

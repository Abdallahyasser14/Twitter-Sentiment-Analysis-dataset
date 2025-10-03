from flask import Flask, request, jsonify
import joblib
import spacy

# Load the saved model, TF-IDF vectorizer, and spaCy model
model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf.pkl')
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Lemmatize and remove stopwords/punctuation
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc 
                                if not token.is_stop and not token.is_punct])
    return lemmatized_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide JSON with "text" field'}), 400
    
    text = data['text']
    # Preprocess text
    processed_text = preprocess_text(text)
    
    text_vector = tfidf_vectorizer.transform([processed_text])
    
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0].max()
    
    if prediction == 0:
        sentiment = "Tweet has negative sentiment"
    elif prediction == 1:
        sentiment = "Tweet has positive sentiment"
    
    return jsonify({
        'text': text,
        'prediction': sentiment,
        'confidence': float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)

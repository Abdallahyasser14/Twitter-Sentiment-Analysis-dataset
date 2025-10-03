from flask import Flask, request, jsonify
import joblib

# Load the saved model and TF-IDF vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide JSON with "text" field'}), 400
    
    text = data['text']
    text_vector = tfidf_vectorizer.transform([text])
    
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

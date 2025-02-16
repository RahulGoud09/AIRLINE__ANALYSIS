from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    """Preprocess input text (e.g., lowercasing, removing special characters)."""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text

def predict_sentiment(text):
    """Vectorize input text and predict sentiment using the trained model."""
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    return prediction

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle review submission and predict sentiment."""
    text = request.form.get('review', '').strip()
    
    if not text:
        return render_template('index.html', error="Please enter a review before submitting.")

    prediction = predict_sentiment(text)
    
    sentiment = 'Neutral 😐'
    if prediction == 1:
        sentiment = 'Positive 😊'
    elif prediction == -1:
        sentiment = 'Negative 😞'

    return render_template('predict.html', prediction_result=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)

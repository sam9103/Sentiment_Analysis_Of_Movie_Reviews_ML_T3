from flask import Flask, render_template, request
from waitress import serve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

app = Flask(__name__)


model = load_model('saved_model/lstm_model.keras')
with open('saved_model/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

maxlen = 100

def validate_text(input_text):
    """
    Validates the input text to ensure it contains only
    letters, spaces, and basic punctuation.
    """
    pattern = re.compile(r"^[a-zA-Z\s.,!?']+$")  
    return bool(pattern.match(input_text))

@app.route('/', methods=['GET', 'POST'])

def home():
    sentiment = None
    error_message = None

    if request.method == 'POST':
        input_text = request.form.get('review')

        if not input_text:
            error_message = "Review cannot be empty."
        elif not validate_text(input_text):
            error_message = "Review contains invalid characters. Please use only letters, spaces, and basic punctuation."
        else:
            # If valid input
            sequence = tokenizer.texts_to_sequences([input_text])
            padded_sequence = pad_sequences(sequence, maxlen=maxlen)

            prediction = model.predict(padded_sequence)[0][0]
            sentiment = 'Positive' if prediction > 0.5 else 'Negative'

    return render_template('index.html', sentiment=sentiment, error_message=error_message)

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)
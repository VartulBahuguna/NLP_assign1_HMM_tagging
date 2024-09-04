from model import HMM_model

# main.py

import pickle

from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS
import re

# Load the model from data.pkl
with open('data.pkl', 'rb') as file:
    hmm_model = pickle.load(file)
def add_spaces_around_punctuation(text):
    # Add spaces around punctuation
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence', '')
    s = add_spaces_around_punctuation(sentence).split(" ")
    s=["<start>"]+s+["<end>"]
    
    #Use the model to get the prediction
    #Replace the following line with your model's prediction logic
    result = hmm_model.HMM_logic(s)
    formatted_result = [(word, tag) for word, tag in result]
    print(formatted_result)

    return jsonify({'result': formatted_result})


@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(port=5000)

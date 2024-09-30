from model import HMM_model
import numpy as np
import nltk
# import matplotlib.pyplot as plt
nltk.download('brown')
nltk.download('universal_tagset')
import pickle

# from flask import Flask, request, jsonify,send_from_directory
# from flask_cors import CORS
import re


from nltk.corpus import brown
#add start and end tags:
def add_start_end_tag(sentences_original):
    sentences=[""]*len(sentences_original)
    for i in range(len(sentences_original)):
        sentences[i]=[("<start>","<start>")]+sentences_original[i]+[("<end>","<end>")]
    return sentences

def remove_tags(sentence):
    new_sentence=[]
    for i in sentence:
        new_sentence.append(i[0])

    # print(new_sentence)

    return new_sentence

def only_tags(sentence):
    new_sentence=[]
    for i in sentence:
        new_sentence.append(i[1])

    # print(new_sentence)

    return new_sentence

#taking data from corpus

# sentences_original = add_start_end_tag(sentences_original)

# main.py


# Load the model from data.pkl
with open('data.pkl', 'rb') as file:
    hmm_model = pickle.load(file)

def add_spaces_around_punctuation(text):
    # Add spaces around punctuation
    text = re.sub(r'([.,!?;:()])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text
# app = Flask(__name__)
# CORS(app) 

# @app.route('/predict', methods=['POST'])
def HMM_predict(sentence):
    # data = request.json
    # sentence = data.get('sentence', '')
    s = add_spaces_around_punctuation(sentence).split(" ")
    # s=["<start>"]+s+["<end>"]
    
    #Use the model to get the prediction
    #Replace the following line with your model's prediction logic
    result = hmm_model.HMM_logic(s)
    formatted_result = [(word, tag) for word, tag in result]
    # print(formatted_result)

    return formatted_result


# @app.route('/')
# def index():
#     return send_from_directory('', 'index.html')

# if __name__ == '__main__':
#     app.run(port=5000)
# Feature extraction functions
def sent2features(sent):
    features = []
    features.append(start_features())
    for i in range(1, len(sent) - 1):
        features.append(word2features(sent, i))
    features.append(end_features())
    return features

def sent2labels(sent):
    return [label for token, label in sent]

def start_features():
    return {'is_start': True}

def end_features():
    return {'is_end': True}

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word': word,
        'is_first': i == 1,
        'is_last': i == len(sent) - 2,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
        'capitals_inside': word[1:].lower() != word[1:]
    }
    return features

# Helper functions to get transition and emission scores (same as before)
def get_transition_scores(crf_model):
    labels = crf_model.classes_
    transition_scores = np.zeros((len(labels), len(labels)))

    for (from_label, to_label), weight in crf_model.transition_features_.items():
        i = labels.index(from_label)
        j = labels.index(to_label)
        transition_scores[i, j] = weight

    return transition_scores

def get_state_scores(crf_model, feature_sequence):
    # Open the trained model

    # Get labels and pre-compute the label indices
    labels = crf_model.classes_
    label_indices = {label: idx for idx, label in enumerate(labels)}

    # Initialize state scores matrix
    state_scores = np.zeros((len(feature_sequence), len(labels)))

    # Extract all state features once
    state_features = crf_model.state_features_
        # 'bias': 1.0,
        # 'word': word,
        # 'is_first': i == 1,  # Adjusted because we have <START> tag
        # 'is_last': i == len(sent) - 2,  # Adjusted because of <END> tag
        # 'is_capitalized': word[0].upper() == word[0],
        # 'is_all_caps': word.upper() == word,
        # 'is_all_lower': word.lower() == word,
        # 'prefix-1': word[0],
        # 'prefix-2': word[:2],
        # 'prefix-3': word[:3],
        # 'suffix-1': word[-1],
        # 'suffix-2': word[-2:],
        # 'suffix-3': word[-3:],
        # 'has_hyphen': '-' in word,
        # 'is_numeric': word.isdigit(),
        # 'capitals_inside': word[1:].lower() != word[1:]

    boolean=['is_first','is_last','is_capitalized','is_all_caps','is_all_lower','has_hyphen','is_numeric','capitals_inside','is_start','is_end']

    # Iterate through each token's features in the sequence
    for t, features in enumerate(feature_sequence):
        # Create a counter for features
        for feat in features.keys():

            if feat in boolean:
                for label in labels :
                    if(features[feat]):
                      weight = state_features.get((feat, label), 0)
                      state_scores[t, label_indices[label]] += weight

            elif feat=="bias":
              for label in labels :

                    weight = state_features.get((feat, label), 0)
                    state_scores[t, label_indices[label]] += weight

            else:
              for label in labels :

                    weight = state_features.get((feat+":"+features[feat], label), 0)
                    state_scores[t, label_indices[label]] += weight

    return state_scores

def custom_predict(test_sentence, crf):
    features = sent2features(test_sentence)
    transmission = get_transition_scores(crf)
    emission = get_state_scores(crf, features)

    l = {}
    l[0] = 1
    seq = [["<START>"]]
    labels = crf.classes_
    for k in range(1, len(features)):
        l_new = []
        l_dict = {}
        c = 0
        for i in l.keys():
            for j in range(len(labels)):
                prev = labels.index(seq[i][-1])
                prob = transmission[prev][j] + emission[k][j]
                l_new.append(seq[i] + [labels[j]])
                l_dict[c] = l[i] + prob
                c += 1
        l_dict = dict(sorted(l_dict.items(), key=lambda item: item[1], reverse=True)[:4])
        seq = []
        l = {}
        c = 0
        for b in l_dict.keys():
            seq.append(l_new[b])
            l[c] = l_dict[b]
            c += 1
    #print(seq[0],test_sentence)
    return seq[0]
def predict_sentence(crf, sentence):
    # Prepare the sentence in the required format
    formatted_sentence = [(word, "") for word in sentence]

    # Predict tags using the custom Viterbi-like method
    predicted_tags = custom_predict(formatted_sentence, crf)

    # Return a list of (word, tag) tuples
    return predicted_tags#list(zip(sentence.split(" "), predicted_tags))

sentences_original = brown.tagged_sents(tagset='universal')

with open('crf_model.pkl', 'rb') as f:
    crf = pickle.load(f)

crf_predicted_data = []
HMM_predicted_data = []

add_start_end_tag(sentences_original)
# for i in range(0, len(crf_sentences_original)):
#     crf_predicted_data.append(predict_sentence(crf, remove_tags(crf_sentences_original[i])))

# sentence = "<START> I wish to have someone in my life who will encourage me in my efforts . <END>"
# prediction = predict_sentence(crf, sentence)
# print("Predicted Tags:", crf_predicted_data)



# HMM_sentences_original[0]
# HMM_predicted_data.append(only_tags(predict(" ".join(remove_tags(sentences_original[0])))))

for i in range(0, len(sentences_original)):
    hmm_sent = " ".join(remove_tags(sentences_original[i]))
    crf_sent = remove_tags(sentences_original[i])
    # if(len(hmm_sent) == len(crf_sent)):
    HMM_predicted_data.append(only_tags(HMM_predict(hmm_sent)))
    crf_predicted_data.append(predict_sentence(crf, crf_sent))
# print(HMM_predicted_data[0:10])

with open('comparision.pkl', 'wb') as f:
    pickle.dump((HMM_predicted_data, crf_predicted_data), f)
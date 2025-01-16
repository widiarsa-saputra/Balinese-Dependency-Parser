from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import fasttext
import pycrfsuite
from spacy import displacy
import tempfile
import os

app = Flask(__name__)

# FastText model
emb_model = fasttext.load_model('model_fasttext.bin')

class EdgeScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EdgeScorer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)         # Output layer
        self.sigmoid = nn.Sigmoid()                # Output: Probabilitas (0-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    

def word_embedding(word):
    """
    fungsi ini digunakan untuk mendapatkan vektor embedding dari suatu kata
    """
    return f'{list(emb_model.get_word_vector(word))}'

def word_embedding_ns(word):
    return emb_model.get_word_vector(word)


def find_score(sentence, model):
    sentence_split = sentence.split()
    edge = []
    scores = []

    for index_i, word_i in enumerate(sentence_split):
        for index_j, word_j in enumerate(sentence_split):
            if index_i != index_j:
                emb_i = word_embedding_ns(word_i)
                emb_j = word_embedding_ns(word_j)
                input_vec = torch.tensor(np.concatenate((emb_i, emb_j)), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    prob = model(input_vec).item()
                edge.append([word_i, word_j])
                scores.append(prob)
    return scores, edge

def edge_score_map(model, sentence):
    # Inisialisasi model
    input_x = []

    scores, edge = find_score(sentence, model)

    for i in range(len(scores)):
        input_x.append((edge[i][0], edge[i][1], scores[i]))

    return input_x


def define_input(sentence):
    input_dim = 200
    hidden_dim = 128
    model = EdgeScorer(input_dim, hidden_dim)
    model.load_state_dict(torch.load('edge_scorer.pth'))
    model.eval()

    input_x = edge_score_map(model, sentence)
    return input_x


def mst_parser(sentence):
    edges = define_input(sentence)

    # Membuat graf terarah
    G = nx.DiGraph()
    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    # Mencari MST menggunakan Chu-Liu/Edmonds
    mst = nx.minimum_spanning_arborescence(G)

    # Menampilkan hasil MST
    # for u, v, weight in mst.edges(data=True):
    #     print(f'{v} -> {u} dengan bobot {weight["weight"]}')
    return mst


def construct_head(sentence, sentence_split):
    edge_score = list(mst_parser(sentence).edges)
    
    tail = [x[1] for x in edge_score]
    dependence_word = []

    for index, value in enumerate(sentence_split):
        if value in tail:
            head = edge_score[int(tail.index(value))]
            dependence_word.append([value, head[0]])
        else:
            dependence_word.append([value])
            
    head_output = []

    for dependence in dependence_word:
        if len(dependence) == 1:
            head_output.append([dependence[0], 0])
        else:
            head_output.append([dependence[0], sentence_split.index(dependence[1])])
    return head_output


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'index=' + str(i),
        'head=' + str(sent[i][2])
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def word2features_pos(sent, i):
    """
    fungsi ini digunakan untuk membentuk feature input pada pos
    sent => list kata yang berisikan [word]
    i => index dari kata saat ini
    model => merupakan model word_embedding yang dipanggil untuk mendapatkan vektor kata
    """
    word = sent[i]
    features = {
        'bias':1.0,
        'word': word,
        'emb_word': word_embedding(word), 
        'is_first' : i == 0,
        'is_last' : i == len(sent)-1,
        'is_title' : word[0].upper() == word[0],
        'isupper' : word.upper() == word,
        'islower' : word.lower() == word,
        'prefix-1' : word[0],
        'prefix-2' : word[:2],
        'suffix-1' : word[-1],
        'suffix-2' : word[-2:],

        'prev_word-1': '' if i == 0 else (sent[i-1]),
        'prev_word-1_prefix-1' : '' if i == 0 else (sent[i-1][0]),
        'prev_word-1_prefix-2' : '' if i == 0 else (sent[i-1][:2]),
        'prev_word-1_suffix-1' : '' if i == 0 else (sent[i-1][-1]),
        'prev_word-1_suffix-2' : '' if i == 0 else (sent[i-1][-2:]),

        'prev_word-2' : (sent[i-2][0]) if i > 1 else '',
        'prev_word-2_prefix-1' : (sent[i-2][0]) if i > 1 else '',
        'prev_word-2_prefix-2' : (sent[i-2][:2]) if i > 1 else '',
        'prev_word-2_suffix-1' : (sent[i-2][-1]) if i > 1 else '',
        'prev_word-2_suffix-2' : (sent[i-2][-2:]) if i > 1 else '',

        'next_word-1' : '' if i == len(sent)-1 else (sent[i+1][0]),
        'next_word-1_prefix-1' : '' if i == len(sent)-1 else (sent[i+1][0]),
        'next_word-1_prefix-2' : '' if i == len(sent)-1 else (sent[i+1][:2]),
        'next_word-1_suffix-1' : '' if i == len(sent)-1 else (sent[i+1][-1]),
        'next_word-1_suffix-2' : '' if i == len(sent)-1 else (sent[i+1][-2:]),

        'next_word-2' : (sent[i+2][0]) if i < len(sent)-2 else '',
        'next_word-2_prefix-1' : (sent[i+2][0]) if i < len(sent)-2 else '',
        'next_word-2_prefix-2' : (sent[i+2][:2]) if i < len(sent)-2 else '',
        'next_word-2_suffix-1' : (sent[i+2][-1]) if i < len(sent)-2 else '',
        'next_word-2_suffix-2' : (sent[i+2][-2:]) if i < len(sent)-2 else '',

        'emb_prev_word-1': '' if i == 0 else word_embedding(sent[i-1][0]),
        'emb_prev_word-1_prefix-2' : '' if i == 0 else word_embedding(sent[i-1][:2]),
        'emb_prev_word-1_suffix-2' : '' if i == 0 else word_embedding(sent[i-1][-2:]),

        'emb_prev_word-2' : word_embedding(sent[i-2][0]) if i > 1 else '',
        'emb_prev_word-2_prefix-2' : word_embedding(sent[i-2][:2]) if i > 1 else '',
        'emb_prev_word-2_suffix-2' : word_embedding(sent[i-2][-2:]) if i > 1 else '',

        'emb_next_word-1' : '' if i == len(sent)-1 else word_embedding(sent[i+1][0]),

        'emb_next_word-1_prefix-2' : '' if i == len(sent)-1 else word_embedding(sent[i+1][:2]),
        'emb_next_word-1_suffix-2' : '' if i == len(sent)-1 else word_embedding(sent[i+1][-2:]),

        'emb_next_word-2' : word_embedding(sent[i+2][0]) if i < len(sent)-2 else '',
        'emb_next_word-2_prefix-2' : word_embedding(sent[i+2][:2]) if i < len(sent)-2 else '',
        'emb_next_word-2_suffix-2' : word_embedding(sent[i+2][-2:]) if i < len(sent)-2 else '',

    }
                
    return features


def sent2features_pos(sent):
    """
    fungsi ini merupakan fungsi untuk memanggil fungsi feature dan mengembalikan feature kata pada kalimat
    """
    return [word2features_pos(sent, i) for i in range(len(sent))]


def make_feature2predict(sentence, pos_sentence):
    feature = []
    pos = pos_sentence
    for index, word in enumerate(sentence.split()):
        feature.append([word, pos[index]])
    return feature


def feature_construct(sentence, pos_sentence):
    head_features = construct_head(sentence, sentence.split())
    features = make_feature2predict(sentence, pos_sentence)
    for index, value in enumerate(features):
        features[index].append(head_features[index][1])
    return features

def construct_displacy(head_features, edge_label, pos_sentence):
    displacy_words = []
    displacy_arcs = []

    words, heads = zip(*head_features)
    labels = edge_label
    for index in range(len(heads)):
        if heads[index] != 0:
            dir = "left" if index > heads[index] else "right"
            displacy_arcs.append({
                "start" : index,
                "end" : heads[index],
                "label" : str(labels[index]),
                "dir" : str(dir)
            })
        displacy_words.append({
                "text" : str(words[index]),
                "tag" : str(pos_sentence[index])
            })
    return {
        "words" : displacy_words,
        "arcs" : displacy_arcs
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        if sentence:
            tagger = pycrfsuite.Tagger()
            tagger.open('label_scorer.crfsuite')

            tagger_pos = pycrfsuite.Tagger()
            tagger_pos.open('person_balinese_pos_2.crfsuite')
            # Run model for feature extraction and dependency parsing
            pos_sentence = tagger_pos.tag(sent2features_pos(sentence.split()))
            output_label = tagger.tag(sent2features(feature_construct(sentence, pos_sentence)))
            head_features = construct_head(sentence, sentence.split())
            print(f"Head features: {head_features}")
            
            # Visualize using displacy
            # with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
            #     data_visual = construct_displacy(head_features, output_label, pos_sentence)
            #     html = displacy.render(data_visual, style="dep", manual=True, page=False, minify=True, options={"page": False, "width": "100%"})
            #     tmpfile.write(html)
            #     tmpfile.close()

            #     with open(tmpfile.name, "r", encoding="utf-8") as f:
            #         html_content = f.read()
            html_content = "Visualization is disabled for debugging."

            # Return the rendered page with the dependency parse visual
            return render_template('index.html', sentence=sentence, visualization=html_content)

    return render_template('index.html', sentence='', visualization='')

if __name__ == "__main__":
    app.run(debug=True)

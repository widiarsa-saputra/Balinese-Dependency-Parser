import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import fasttext
import pycrfsuite
import streamlit as st
import tempfile
from bs4 import BeautifulSoup
from spacy import displacy
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="expanded")



emb_model = fasttext.load_model('model_fasttext.bin')

st.markdown("""
    <style>
        .stMarkdown {
            width: fit-content !important;
            block-size: fit-content !important;
            text-align: center;
        }
        .stButton {
            display: flex;
            justify-content: center;    
        }
        .stTextInput {
            width: 800px;    
        }
        .element-container {
            display: flex;
            justify-content: center;    
        }
        .stTextInput>div>div>input {
            padding: 10px;
            border-radius: 5px;
            background-color: white;
            color: black;
            border-radius: 3px;
            border: 2px solid #90b7e0;
            text-align: center;  /* Mengatur teks agar berada di tengah horizontal */
            vertical-align: middle;  /* Mengatur teks agar berada di tengah vertikal */
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)


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


st.markdown("""
    <style>
        .block-container {
                
        }
    </style>
""", unsafe_allow_html=True)

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

def visualize(data):
    """
    Fungsi ini digunakan untuk visualisasi ke web page menggunakan displacy.
    Disesuaikan untuk Streamlit dengan menggunakan `st.components.v1.html()`.
    """

    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
        html = displacy.render(data, style="dep", manual=True, page=False, minify=True, options={"page": False, "width": "100%"})

        tmpfile.write(html)
        tmpfile.close()

        # Membaca file HTML dan menambahkan CSS untuk mengatasi masalah lebar
        with open(tmpfile.name, "r", encoding="utf-8") as f:
            html_content = f.read()

            # Menambahkan CSS untuk mengatur lebar visualisasi
            html_content = f'''
            <style>
                body {{
                    width: 100% !important;
                    margin: 0;
                    padding: 0;
                }}
                .displacy {{
                    width: 100% !important;
                    margin: 0;   /* Menghilangkan margin yang membuat pemusatan */
                    padding: 0;
                    display: block;
                }}
                .displacy .container {{
                    width: 100% !important;
                    margin: 0 auto;
                    display: block;
                }}
            </style>
            {html_content}
            '''

            # Menampilkan HTML dalam Streamlit menggunakan components dengan lebar lebih besar
            components.html(html_content, height=800, width=1200)  # Menyesuaikan lebar

def main():
    st.title("BALINESE DEPENDENCY PARSER")
    tagger = pycrfsuite.Tagger()
    tagger.open('label_scorer.crfsuite')

    tagger_pos = pycrfsuite.Tagger()
    tagger_pos.open('person_balinese_pos_2.crfsuite')

    # Input text from the user
    sentence = st.text_input('',placeholder="Masukkan Teks")


    # Create a container for the button on the right
    
    if sentence:
        button_placeholder = st.empty()
        button = button_placeholder.button("show")
        pos_sentence = tagger_pos.tag(sent2features_pos(sentence.split()))
        output_label = tagger.tag(sent2features(feature_construct(sentence, pos_sentence)))
        head_features = construct_head(sentence, sentence.split())
        if button:
            word, head = zip(*head_features)
            head_word = [word[int(i)] for i in head]
            df = pd.DataFrame({
                    'word' : word,
                    'head' : head_word,
                    'label' : output_label
                })
            st.markdown("""
                            <style>
                                .css-1ec096l tbody tr th, .dataframe tbody tr td {
                                    font-size: 20px;
                                    padding: 20px;
                                }
                            </style>
                        """, unsafe_allow_html=True)

            st.table(df)
            # for i in range(len(head_features)):
            #     st.write(f"{word[i]} > {head_word[i]} ({output_label[i]})")
            button_placeholder.empty()
            button = button_placeholder.button("displacy")
        else: 
            # Predict entities from the input text
            with st.spinner("Memproses... Harap tunggu"):
                # st.write(pos_sentence)
                # st.write(output_label)
                # st.write(head_features)
                try:
                    data_visual = construct_displacy(head_features, output_label, pos_sentence)
                    if data_visual:
                        st.write(" ")
                        visualize(data_visual)
                    else:
                        st.write("Tidak ada entitas yang terdeteksi.")
                except ValueError:
                    st.write("error displacy")

if __name__ == "__main__":
    main()



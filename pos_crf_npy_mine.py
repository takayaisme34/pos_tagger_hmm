from collections import defaultdict
import json
import pickle
import numpy as np

with open("crf_index.json", "r", encoding="utf-8") as f:
    crf_index = json.load(f)

with open('crf_model.pkl', "rb") as f:
    params = pickle.load(f)


#token = (surface, reading, lemma, pos, pos_subcategory, conjugation_type, conjugation_form)
vocab2i = {tuple(token):int(i) for i, token in  crf_index['vocab'].items()}
surface2i = crf_index['surface']
max_len = crf_index['max_len']
lemma2i = crf_index['lemma']
pos2i = crf_index['pos']
pos_subcat2i = crf_index['pos_subcat']
conj_type2i = crf_index['conj_type']
conj_form2i = crf_index['conj_form']

BOS = EOS = len(pos2i)

transition_matrix = params['transition']
emission = params['emission']
surface2tups = defaultdict(list)
for tup in vocab2i:
    surface2tups[tup[0]].append(tup)
max_word_len = max(len(s) for s in surface2tups)

def emission_score(node):
    return (emission['lemma'][lemma2i[node['lemma']]] +
            emission['pos'][pos2i[node['pos']]] +
            emission['pos_subcat'][pos_subcat2i[node['pos_subcat']]] +
            emission['conj_type'][conj_type2i[node['conj_type']]] +
            emission['conj_form'][conj_form2i[node['conj_form']]])

def transition_score(prev_pos, curr_pos):
    return transition_matrix[pos2i.get(prev_pos, BOS), pos2i[curr_pos]]

def build_lattice(x):
    lattice = [[] for _ in range(len(x) + 1)]   # lattice[end]; index 0 = BOS
    lattice[0].append({'pos': None, 'score': 0.0, 'best_prev': None})
    for end in range(1, len(x) + 1):
        for start in range(max(0, end - max_word_len), end):
            for tup in surface2tups.get(x[start:end], []):
                keys = ['surface','reading','lemma','pos','pos_subcat','conj_type','conj_form']
                lattice[end].append(dict(zip(keys, tup)) | {'start': start, 'end': end})
    return lattice

def viterbi(lattice):
    for end in range(1, len(lattice)): # 1 skips bos
        for node in lattice[end]:
            prevs = lattice[node['start']]
            if not prevs:
                node['score'] = -np.inf; node['best_prev'] = None; continue
            candidates = [
                prev_node['score'] + transition_score(prev_node['pos'], node['pos']) 
                for prev_node in prevs
            ]
            i = int(np.argmax(candidates))
            node['score'] = candidates[i] + emission_score(node)
    node = max(lattice[-1], key=lambda n: n['score'])   
    path = []
    while True:
        path.append(node)
        node_start = node['start']
        prev_node = max(lattice[node_start], key=lambda n: n['score'])
        if node_start == 0:
            break
        node = prev_node  
    return path[::-1]

def tokenize(x):
    return viterbi(build_lattice(x))

def tokenize_and_print(x):
    for t in tokenize(x):
        print(t["surface"], t['reading'], sep="\t")


if __name__ == "__main__":
    x1 = "私は東京都に住む大学院生である。"
    x2 = "私は東大阪に住む大学院生である。"

    tokenize_and_print(x1)
    tokenize_and_print(x2)
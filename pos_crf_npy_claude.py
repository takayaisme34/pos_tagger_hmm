from collections import defaultdict
import jax.numpy as jnp
from jax import nn
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
surface2tups = defaultdict(list)
for tup in vocab2i:
    surface2tups[tup[0]].append(tup)
max_word_len = max(len(s) for s in surface2tups)

transition_matrix = params['transition']
emission = params['emission']

#  training uses BOS = EOS = len(pos2i) (the extra last row/column of the
# transition matrix). The BOS node has pos None, so pos2i.get(pos, BOS) maps it there.
BOS = EOS = len(pos2i)

def build_lattice(x):
    lattice = [[] for _ in range(len(x) + 1)]   # lattice[end]; index 0 = BOS
    #  put a BOS node in row 0 so words starting at 0 have a predecessor
    lattice[0].append({'pos': None, 'start': 0, 'end': 0, 'score': 0.0, 'best_prev': None})
    for end in range(1, len(x) + 1):

        for start in range(max(0, end - max_word_len), end):

            for tup in surface2tups.get(x[start:end], []):
                #  no special initial score; every node is scored from its predecessors in viterbi
                keys = ['surface','reading','lemma','pos','pos_subcat','conj_type','conj_form']
                lattice[end].append(dict(zip(keys, tup)) | {'start': start, 'end': end, 'score': -np.inf, 'best_prev': None})
    #  no lattice.pop(0); lattice[node['start']] must be the nodes ending where this node starts
    return lattice

def viterbi(lattice):
    for i, end_position in enumerate(lattice):
        if i == 0:
            continue
        for node in end_position:
            node_start = node['start']
            #  skip unreachable predecessors (no dictionary path up to them)
            previous_nodes = [p for p in lattice[node_start] if np.isfinite(p['score'])]
            if not previous_nodes:
                continue

            now_pos = node['pos']
            #  choose the predecessor by score + transition, not score alone
            candidates = [
                prev_node['score'] + transition_matrix[pos2i.get(prev_node['pos'], BOS), pos2i[now_pos]]
                for prev_node in previous_nodes
            ]
            best = int(np.argmax(candidates))
            best_prev_node = previous_nodes[best]
            emission_score = (
                emission['token'][vocab2i[tuple(node[k] for k in
                    ['surface','reading','lemma','pos','pos_subcat','conj_type','conj_form'])]] +   #  token weight was missing
                emission['lemma'][lemma2i[node['lemma']]] +
                emission['pos'][pos2i[node['pos']]] +
                emission['pos_subcat'][pos_subcat2i[node['pos_subcat']]] +
                emission['conj_type'][conj_type2i[node['conj_type']]] +
                emission['conj_form'][conj_form2i[node['conj_form']]]
            )
            node['score'] = candidates[best] + emission_score
            node['best_prev'] = best_prev_node   #  store a backpointer

    #  pick the best last node (including the EOS transition) and follow backpointers
    finals = [n for n in lattice[-1] if np.isfinite(n['score'])]
    if not finals:
        return None                              # some part of x is not in the vocabulary
    current_node = max(finals, key=lambda n: n['score'] + transition_matrix[pos2i[n['pos']], EOS])

    tagged_sequence = []
    while current_node['best_prev'] is not None:
        tagged_sequence.append(current_node)
        current_node = current_node['best_prev']
    return tagged_sequence[::-1]


if __name__ == "__main__":
    x = "私は東大阪に住む大学院生だ。"
    lattice = build_lattice(x)
    for node in viterbi(lattice):
        print(node['surface'], node['reading'], node['pos'], node['pos_subcat'], sep='\t')
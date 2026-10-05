
import jax.numpy as jnp
from jax import nn
from parse_knp import iterate_knp_sentences
import json
import os
from tqdm import tqdm


if os.path.exists("kwdlc.jsonl"):
    kwdlc = []
    with open("kwdlc.jsonl", "r", encoding="utf-8")as f:
        for line in f:
            kwdlc.append(json.loads(line))
else:
    kwdlc = []
    with open("kwdlc.jsonl", "w", encoding="utf-8") as f:
        for sentence in iterate_knp_sentences():
            f.write(json.dumps(sentence)+"\n")
            kwdlc.append(sentence)

with tqdm(desc="creating vocab", total=len(kwdlc)) as pbar:
    vocab = []
    vocab2i = {}
    #('おり', 'おり', 'おる', '接尾辞', '動詞性接尾辞', '子音動詞ラ行', '基本連用形')
    for sentence in kwdlc:
        for token in sentence:
            tup = tuple(token.values())
            if tup in vocab:
                continue
            vocab.append(tup)
        pbar.update(1)


vocab2i = {token:i for i, token in enumerate(sorted(vocab))}
#max_len = max(len(w[0]) for w in vocab)
pos_tags = set()
lemmas = set()
pos_subcategories = set()
conjugation_forms = set()
conjugation_types = set()
with tqdm(desc="creating params", total=len(vocab)) as pbar:
    for tup in vocab:
        pos_tags.add(tup[3])
        lemmas.add(tup[2])
        pos_subcategories.add(tup[4])
        conjugation_types.add(tup[5])
        conjugation_forms.add(tup[6])
        pbar.update(1)

pos2i = {pos:i for i, pos in enumerate(sorted(pos_tags))}
lemma2i = {lemma:i for i, lemma in enumerate(sorted(lemmas))}
pos_subcat2i = {pos_subcat:i for i, pos_subcat in enumerate(sorted(pos_subcategories))}
conj_form2i = {conj_form:i for i , conj_form in enumerate(sorted(conjugation_forms))}
conj_type2i = {conj_type:i for i, conj_type in enumerate(sorted(conjugation_types))}


params = {
    "transition":jnp.zeros((len(pos_tags)+1,len(pos_tags)+1)),
    "emission":{
        "pos":jnp.zeros((len(vocab), len(pos_tags))),
        "lemma":jnp.zeros(len(lemmas)),
        "pos_subcat":jnp.zeros((len(vocab), len(pos_subcategories))),
        "conj_type":jnp.zeros((len(vocab), len(conjugation_types))),
        "conj_form":jnp.zeros((len(vocab), len(conjugation_forms))),
    }
}

def score_emission(params, token):
    pass

def score_transition(params, node, prev_node):
    pass

token = ('おり', 'おり', 'おる', '接尾辞', '動詞性接尾辞', '子音動詞ラ行', '基本連用形')

print(lemmas)
print("\n\n")
print(pos_subcategories)
print("\n\n")
print(conjugation_types)
print("\n\n")
print(conjugation_forms)
print("\n\n")


exit()

def encode(x):
    return jnp.array([vocab2i[token] for token in x])

def build_lattice(x):
    lattice = [[] for _ in range(len(x)+1)]
    for end in range(len(x)+1):
        for token in vocab:
            surface = token[0]
            str_extract = x[end-len(surface):end]
            if str_extract != surface:
                continue
            lattice[end].append( {
                "surface": token[0], "reading": token[1], "lemma": token[2],
                "pos":token[3], "pos_subcategory":token[4], "conjugation_type": token[5],
                "conjugation_form":token[6],
                "start": end-len(surface), "end": end, "alpha":0
            })
    return lattice

def calculate_log_z(params, lattice, scores):
    for end_position in lattice:                      # in order of end position
        for node in end_position:
            unigram_score = score_emission(params,)
            if node["start"] == 0:
                node["alpha"] = s                 # predecessor is BOS, log α(BOS) = 0
            else:
                prev = jnp.array([n["alpha"] for n in lattice[node["start"]]])
                node["alpha"] = nn.logsumexp(prev) + s
    return nn.logsumexp(jnp.array([n["alpha"] for n in lattice[-1]]))



if __name__ == "__main__":
    
    x = "私は東京都に住む大学院生だ。"
    #x_encoded = encode(x)
    lattice = build_lattice(x)
    print(lattice)
    #calculate_z(lattice)
    #print(calculate_log_z(lattice, gold_scores))





import jax
import jax.numpy as jnp
from jax import nn
from collections import defaultdict
from parse_knp import iterate_knp_sentences
import json
import os
from tqdm import tqdm


if os.path.exists("kwdlc.jsonl"):
    kwdlc = []
    with open("kwdlc.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            kwdlc.append(json.loads(line))
else:
    kwdlc = []
    with open("kwdlc.jsonl", "w", encoding="utf-8") as f:
        for sentence in iterate_knp_sentences():
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
            kwdlc.append(sentence)

# each token: (surface, reading, lemma, pos, pos_subcat, conj_type, conj_form)
# e.g.        ('おり', 'おり', 'おる', '接尾辞', '動詞性接尾辞', '子音動詞ラ行', '基本連用形')
sentences = [[tuple(token.values()) for token in sentence] for sentence in kwdlc]

vocab = set()
for sentence in tqdm(sentences, desc="creating vocab"):
    vocab.update(sentence)                      # a set: `in list` would be O(n) per check
vocab = sorted(vocab)
vocab2i = {token: i for i, token in enumerate(vocab)}

surface2tokens = defaultdict(list)              # surface -> all dictionary entries with that surface
for token in vocab:
    surface2tokens[token[0]].append(token)
max_len = max(len(s) for s in surface2tokens)

lemma2i      = {v: i for i, v in enumerate(sorted({t[2] for t in vocab}))}
pos2i        = {v: i for i, v in enumerate(sorted({t[3] for t in vocab}))}
pos_subcat2i = {v: i for i, v in enumerate(sorted({t[4] for t in vocab}))}
conj_type2i  = {v: i for i, v in enumerate(sorted({t[5] for t in vocab}))}
conj_form2i  = {v: i for i, v in enumerate(sorted({t[6] for t in vocab}))}
BOS = EOS = len(pos2i)                          # extra transition index for the sentence boundary

# one weight per feature value; a node's emission score is the sum of the weights of its features
params = {
    "transition": jnp.zeros((len(pos2i) + 1, len(pos2i) + 1)),   # [prev pos or BOS, pos or EOS]
    "emission": {
        "token":      jnp.zeros(len(vocab)),       # the full 7-tuple
        "lemma":      jnp.zeros(len(lemma2i)),
        "pos":        jnp.zeros(len(pos2i)),
        "pos_subcat": jnp.zeros(len(pos_subcat2i)),
        "conj_type":  jnp.zeros(len(conj_type2i)),
        "conj_form":  jnp.zeros(len(conj_form2i)),
    },
}


def make_node(token, start):
    return {
        "surface": token[0], "reading": token[1], "lemma": token[2],
        "pos": token[3], "pos_subcategory": token[4], "conjugation_type": token[5],
        "conjugation_form": token[6], "id": vocab2i[token],
        "start": start, "end": start + len(token[0]),
    }

def score_emission(params, node):
    e = params["emission"]
    return (e["token"][node["id"]]
            + e["lemma"][lemma2i[node["lemma"]]]
            + e["pos"][pos2i[node["pos"]]]
            + e["pos_subcat"][pos_subcat2i[node["pos_subcategory"]]]
            + e["conj_type"][conj_type2i[node["conjugation_type"]]]
            + e["conj_form"][conj_form2i[node["conjugation_form"]]])

def score_transition(params, node, prev_node):
    """prev_node=None means BOS, node=None means EOS."""
    i = BOS if prev_node is None else pos2i[prev_node["pos"]]
    j = EOS if node is None else pos2i[node["pos"]]
    return params["transition"][i, j]

def build_lattice(x):
    lattice = [[] for _ in range(len(x) + 1)]
    for end in range(1, len(x) + 1):
        for start in range(max(0, end - max_len), end):
            if start > 0 and not lattice[start]:
                continue                        # no path reaches `start`, so skip
            for token in surface2tokens.get(x[start:end], []):
                lattice[end].append(make_node(token, start))
    return lattice

def calculate_log_z(params, lattice):
    alpha = [[] for _ in lattice]               # alpha[end][k] = log alpha of lattice[end][k]
    for end in range(1, len(lattice)):          # in order of end position
        for node in lattice[end]:
            if node["start"] == 0:              # predecessor is BOS, log alpha(BOS) = 0
                a = score_transition(params, node, None)
            else:
                prev = lattice[node["start"]]
                a = nn.logsumexp(jnp.array([
                    alpha[node["start"]][k] + score_transition(params, node, p)
                    for k, p in enumerate(prev)]))
            alpha[end].append(a + score_emission(params, node))
    return nn.logsumexp(jnp.array([
        alpha[-1][k] + score_transition(params, None, n) for k, n in enumerate(lattice[-1])]))

def gold_score(params, sentence):
    """Score of the correct path: same two functions, applied along the gold tokens."""
    total, prev, start = 0.0, None, 0
    for token in sentence:
        node = make_node(token, start)
        total += score_transition(params, node, prev) + score_emission(params, node)
        prev, start = node, node["end"]
    return total + score_transition(params, None, prev)

def nll(params, lattice, sentence):
    return calculate_log_z(params, lattice) - gold_score(params, sentence)


if __name__ == "__main__":
    train = [(build_lattice("".join(t[0] for t in s)), s)
             for s in tqdm(sentences, desc="building lattices")]

    grad_fn = jax.value_and_grad(nll)
    lr = 0.1
    for epoch in range(10):
        total = 0.0
        for lattice, sentence in tqdm(train, desc=f"epoch {epoch}", leave=False):
            loss, grads = grad_fn(params, lattice, sentence)
            params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
            total += loss
        print(f"epoch {epoch}  loss {total / len(train):.4f}")

    x = "私は東京都に住む大学院生だ。"
    print(calculate_log_z(params, build_lattice(x)))   # -inf if some part of x is not in vocab
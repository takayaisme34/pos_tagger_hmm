import json
import os
import random
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, nn
from tqdm import tqdm


# ============================================================ data
def load_kwdlc(path=os.path.join("KWDLC-1.0", "kwdlc.jsonl")):
    if not os.path.exists(path):
        from parse_knp import iterate_knp_sentences
        with open(path, "w", encoding="utf-8") as f:
            for sentence in iterate_knp_sentences():
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# each token: (surface, reading, lemma, pos, pos_subcat, conj_type, conj_form)
sentences = [[tuple(token.values()) for token in s] for s in load_kwdlc()]

vocab = sorted({token for s in sentences for token in s})
vocab2i = {token: i for i, token in enumerate(vocab)}

surface2ids = defaultdict(list)                  # surface -> ids of tokens with that surface
for i, token in enumerate(vocab):
    surface2ids[token[0]].append(i)
max_len = max(len(surface) for surface in surface2ids)


# ============================================================ features
# For each feature, an array that maps token id -> feature value id.
COLUMNS = {"lemma": 2, "pos": 3, "pos_subcat": 4, "conj_type": 5, "conj_form": 6}

feature_ids = {"token": jnp.arange(len(vocab))}
num_values = {"token": len(vocab)}
for name, col in COLUMNS.items():
    values = sorted({token[col] for token in vocab})
    value2i = {v: i for i, v in enumerate(values)}
    feature_ids[name] = jnp.array([value2i[token[col]] for token in vocab])
    num_values[name] = len(values)

POS = feature_ids["pos"]                         # token id -> pos id
BOS = EOS = num_values["pos"]                    # one extra pos id for sentence start/end
NEG = -1e9                                       # "impossible"; -inf would give NaN gradients

params = {
    "transition": jnp.zeros((BOS + 1, BOS + 1)),                      # [pos before, pos after]
    "emission": {name: jnp.zeros(n) for name, n in num_values.items()},
}

def emission(params, tok):
    """Score of each token = sum of its feature weights. tok can have any shape."""
    return sum(params["emission"][name][ids[tok]] for name, ids in feature_ids.items())

def node_pos(tok):
    return POS[tok].at[0].set(BOS)               # node 0 is always BOS


# ============================================================ lattice
# The lattice is a flat list of nodes (token_id, start, end), sorted by end.
# Node 0 is BOS, which ends at position 0.
# Node m can come right before node n  <=>  end[m] == start[n].
def build_lattice(x):
    nodes = [(0, 0, 0)]
    reached = {0}                                # positions that some path from BOS reaches
    for end in range(1, len(x) + 1):
        for start in range(max(0, end - max_len), end):
            if start not in reached:
                continue
            for token_id in surface2ids.get(x[start:end], []):
                nodes.append((token_id, start, end))
                reached.add(end)
    return nodes

def to_arrays(nodes, N):
    """Pad the lattice to N nodes. Padding has end = -1, so nothing ever follows it."""
    tok = np.zeros(N, np.int32)
    start = np.zeros(N, np.int32)
    end = np.full(N, -1, np.int32)
    for n, (t, s, e) in enumerate(nodes):
        tok[n], start[n], end[n] = t, s, e
    return tok, start, end

def round_up(n, m=16):                           # fewer distinct shapes -> fewer recompilations
    return -(-n // m) * m


# ============================================================ CRF
def log_Z(params, tok, start, end, length):
    """Forward algorithm: log of the summed exp-score of every path through the lattice."""
    T, emit, pos = params["transition"], emission(params, tok), node_pos(tok)

    def visit(alpha, n):                         # one node at a time, in order of end
        is_prev = end == start[n]                # nodes that can come right before n
        from_prev = alpha + T[pos, pos[n]]       # alpha(m) + transition(m -> n), for every m
        a = nn.logsumexp(jnp.where(is_prev, from_prev, NEG)) + emit[n]
        return alpha.at[n].set(a), None

    alpha = jnp.full(len(tok), NEG).at[0].set(0.0)               # log alpha(BOS) = 0
    alpha, _ = lax.scan(visit, alpha, jnp.arange(1, len(tok)))
    is_last = end == length                      # nodes that end the sentence
    return nn.logsumexp(jnp.where(is_last, alpha + T[pos, EOS], NEG))

def gold_score(params, gold, n_gold):
    """Score of the correct path. gold = its token ids, padded after n_gold."""
    T = params["transition"]
    pos = POS[gold]
    prev_pos = jnp.concatenate([jnp.array([BOS]), pos[:-1]])
    is_real = jnp.arange(len(gold)) < n_gold
    steps = jnp.where(is_real, emission(params, gold) + T[prev_pos, pos], 0.0)
    return steps.sum() + T[pos[n_gold - 1], EOS]

def nll(params, tok, start, end, length, gold, n_gold):
    return log_Z(params, tok, start, end, length) - gold_score(params, gold, n_gold)

def l2(params):
    return sum(jnp.sum(w ** 2) for w in jax.tree.leaves(params))

@jax.jit
def batch_loss_and_grad(params, batch, lam):
    def objective(p):
        per_sentence = jax.vmap(nll, in_axes=(None, 0, 0, 0, 0, 0, 0))(p, *batch)
        return per_sentence.mean() + 0.5 * lam * l2(p)
    return jax.value_and_grad(objective)(params)


# ============================================================ decoding
@jax.jit
def viterbi_arrays(params, tok, start, end, length):
    """Same as log_Z, but max instead of logsumexp, and remember the best previous node."""
    T, emit, pos = params["transition"], emission(params, tok), node_pos(tok)

    def visit(delta, n):
        is_prev = end == start[n]
        from_prev = jnp.where(is_prev, delta + T[pos, pos[n]], NEG)
        return delta.at[n].set(from_prev.max() + emit[n]), from_prev.argmax()

    delta = jnp.full(len(tok), NEG).at[0].set(0.0)
    delta, back = lax.scan(visit, delta, jnp.arange(1, len(tok)))
    back = jnp.concatenate([jnp.array([0]), back])               # back[n] = best node before n
    is_last = end == length
    return jnp.where(is_last, delta + T[pos, EOS], NEG).argmax(), back

def analyse(params, x):
    nodes = build_lattice(x)
    if not any(e == len(x) for _, _, e in nodes):
        return None                              # x contains a word not in the vocabulary
    tok, start, end = to_arrays(nodes, round_up(len(nodes), 64))
    n, back = viterbi_arrays(params, tok, start, end, len(x))
    n, back, path = int(n), np.asarray(back), []
    while n != 0:                                # walk back until BOS
        path.append(vocab[tok[n]])
        n = back[n]
    return path[::-1]


# ============================================================ training
def make_batches(sentences, batch_size=32):
    examples = []
    for s in tqdm(sentences, desc="building lattices"):
        x = "".join(token[0] for token in s)
        examples.append((len(x), build_lattice(x), [vocab2i[token] for token in s]))
    examples.sort(key=lambda ex: len(ex[1]))     # similar lattice sizes together -> less padding

    batches = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        N = round_up(max(len(nodes) for _, nodes, _ in chunk), 64)
        M = round_up(max(len(gold) for _, _, gold in chunk))
        tok, start, end = zip(*[to_arrays(nodes, N) for _, nodes, _ in chunk])
        gold = np.zeros((len(chunk), M), np.int32)
        for j, (_, _, g) in enumerate(chunk):
            gold[j, :len(g)] = g
        lengths = [length for length, _, _ in chunk]
        n_gold = [len(g) for _, _, g in chunk]
        batches.append(tuple(jnp.array(a) for a in
                             (np.stack(tok), np.stack(start), np.stack(end), lengths, gold, n_gold)))
    return batches


if __name__ == "__main__":
    batches = make_batches(sentences)

    lr = 1.0
    sigma2 = 1.0                                 # prior variance for L2; tune on a dev set
    lam = 1.0 / (sigma2 * len(sentences))

    for epoch in range(10):
        random.shuffle(batches)
        total = 0.0
        for batch in tqdm(batches, desc=f"epoch {epoch}", leave=False):
            loss, grads = batch_loss_and_grad(params, batch, lam)
            params = jax.tree.map(lambda w, g: w - lr * g, params, grads)
            total += float(loss)
        print(f"epoch {epoch}  loss {total / len(batches):.4f}")

    print(analyse(params, "私は東京都に住む大学院生だ。"))
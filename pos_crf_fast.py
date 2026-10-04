import jax
import jax.numpy as jnp
import numpy as np
from jax import nn, lax
from collections import defaultdict
from parse_knp import iterate_knp_sentences
import json
import os
import random
from tqdm import tqdm


kwdlc_path = os.path.join("KWDLC-1.0", "kwdlc.jsonl")
if os.path.exists():
    kwdlc = []
    with open(kwdlc_path, "r", encoding="utf-8") as f:
        for line in f:
            kwdlc.append(json.loads(line))
else:
    kwdlc = []
    with open(kwdlc_path, "w", encoding="utf-8") as f:
        for sentence in iterate_knp_sentences():
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
            kwdlc.append(sentence)

# each token: (surface, reading, lemma, pos, pos_subcat, conj_type, conj_form)
sentences = [[tuple(token.values()) for token in sentence] for sentence in kwdlc]

vocab = set()
for sentence in tqdm(sentences, desc="creating vocab"):
    vocab.update(sentence)
vocab = sorted(vocab)
vocab2i = {token: i for i, token in enumerate(vocab)}

surface2ids = defaultdict(list)                 # surface -> token ids with that surface
for i, token in enumerate(vocab):
    surface2ids[token[0]].append(i)
max_len = max(len(s) for s in surface2ids)

lemma2i      = {v: i for i, v in enumerate(sorted({t[2] for t in vocab}))}
pos2i        = {v: i for i, v in enumerate(sorted({t[3] for t in vocab}))}
pos_subcat2i = {v: i for i, v in enumerate(sorted({t[4] for t in vocab}))}
conj_type2i  = {v: i for i, v in enumerate(sorted({t[5] for t in vocab}))}
conj_form2i  = {v: i for i, v in enumerate(sorted({t[6] for t in vocab}))}
BOS = EOS = len(pos2i)
NEG = -1e9                                      # "impossible" score; -inf would give NaN gradients

# token id -> its 6 feature ids, looked up on the device instead of in Python dicts
FEATURES = ["token", "lemma", "pos", "pos_subcat", "conj_type", "conj_form"]
TOKEN_FEATS = jnp.array([[i, lemma2i[t[2]], pos2i[t[3]], pos_subcat2i[t[4]],
                          conj_type2i[t[5]], conj_form2i[t[6]]] for i, t in enumerate(vocab)])
POS = FEATURES.index("pos")

params = {
    "transition": jnp.zeros((len(pos2i) + 1, len(pos2i) + 1)),
    "emission": {
        "token":      jnp.zeros(len(vocab)),
        "lemma":      jnp.zeros(len(lemma2i)),
        "pos":        jnp.zeros(len(pos2i)),
        "pos_subcat": jnp.zeros(len(pos_subcat2i)),
        "conj_type":  jnp.zeros(len(conj_type2i)),
        "conj_form":  jnp.zeros(len(conj_form2i)),
    },
}


# Lattice as padded arrays. Row e holds up to K nodes that END at character e:
#   tok[e, k]   token id        start[e, k]  where the node starts
#   valid[e, k] real node or padding
# Row 0 holds BOS at k = 0. The predecessors of node (e, k) are exactly row start[e, k].
def build_nodes(x):
    nodes = [[] for _ in range(len(x) + 1)]
    nodes[0] = [(0, 0)]                         # BOS placeholder
    for end in range(1, len(x) + 1):
        for start in range(max(0, end - max_len), end):
            if not nodes[start]:
                continue                        # unreachable from BOS
            for t in surface2ids.get(x[start:end], []):
                nodes[end].append((start, t))
    return nodes

def to_arrays(nodes, L, K):
    tok, start = np.zeros((L + 1, K), np.int32), np.zeros((L + 1, K), np.int32)
    valid = np.zeros((L + 1, K), bool)
    for e, row in enumerate(nodes):
        for k, (s, t) in enumerate(row):
            tok[e, k], start[e, k], valid[e, k] = t, s, True
    return tok, start, valid

def emission(params, feats):                    # feats (..., 6) -> scores (...)
    return sum(params["emission"][f][feats[..., i]] for i, f in enumerate(FEATURES))

def log_Z(params, tok, start, valid, length):
    tok, start, valid = jnp.asarray(tok), jnp.asarray(start), jnp.asarray(valid)
    T = params["transition"]
    feats = TOKEN_FEATS[tok]                                    # (L+1, K, 6)
    emit = emission(params, feats)                              # (L+1, K)
    pos = feats[..., POS].at[0].set(BOS)                        # (L+1, K)
    alpha0 = jnp.full(tok.shape, NEG).at[0, 0].set(0.0)         # log alpha(BOS) = 0

    def step(alpha, e):
        prev_alpha = alpha[start[e]]                            # (K, K): row k = predecessors of node k
        prev_pos = pos[start[e]]                                # (K, K)
        cand = prev_alpha + T[prev_pos, pos[e][:, None]]
        new = nn.logsumexp(cand, axis=1) + emit[e]
        return alpha.at[e].set(jnp.where(valid[e], new, NEG)), None

    alpha, _ = lax.scan(step, alpha0, jnp.arange(1, tok.shape[0]))
    return nn.logsumexp(alpha[length] + T[pos[length], EOS])

def gold_score(params, gold, n_gold):
    T = params["transition"]
    feats = TOKEN_FEATS[gold]                                   # (M, 6)
    mask = jnp.arange(gold.shape[0]) < n_gold
    pos = feats[:, POS]
    prev = jnp.concatenate([jnp.array([BOS]), pos[:-1]])
    return ((emission(params, feats) + T[prev, pos]) * mask).sum() + T[pos[n_gold - 1], EOS]

def nll(params, tok, start, valid, length, gold, n_gold):
    return log_Z(params, tok, start, valid, length) - gold_score(params, gold, n_gold)

@jax.jit
def batch_loss_and_grad(params, batch):
    loss_fn = lambda p: jax.vmap(nll, in_axes=(None, 0, 0, 0, 0, 0, 0))(p, *batch).mean()
    return jax.value_and_grad(loss_fn)(params)

@jax.jit
def viterbi_arrays(params, tok, start, valid, length):
    T = params["transition"]
    feats = TOKEN_FEATS[tok]
    emit = emission(params, feats)
    pos = feats[..., POS].at[0].set(BOS)
    delta0 = jnp.full(tok.shape, NEG).at[0, 0].set(0.0)

    def step(delta, e):
        cand = delta[start[e]] + T[pos[start[e]], pos[e][:, None]]
        new = cand.max(axis=1) + emit[e]
        return delta.at[e].set(jnp.where(valid[e], new, NEG)), cand.argmax(axis=1)

    delta, back = lax.scan(step, delta0, jnp.arange(1, tok.shape[0]))
    return jnp.argmax(delta[length] + T[pos[length], EOS]), back

def viterbi(params, x):
    nodes = build_nodes(x)
    if not nodes[-1]:
        return None                             # some part of x is not in the vocabulary
    L, K = round_up(len(x)), max(len(r) for r in nodes)
    tok, start, valid = to_arrays(nodes, L, K)
    k, back = viterbi_arrays(params, tok, start, valid, len(x))
    k, back, path, e = int(k), np.asarray(back), [], len(x)
    while e > 0:
        path.append(vocab[tok[e, k]])
        e, k = start[e, k], back[e - 1, k]
    return path[::-1]

def round_up(n, m=16):                          # fewer distinct shapes -> fewer recompilations
    return -(-n // m) * m


if __name__ == "__main__":
    # ---- preprocess once: lattices and gold paths as padded arrays ----
    data = []
    for s in tqdm(sentences, desc="building lattices"):
        x = "".join(t[0] for t in s)
        data.append((len(x), build_nodes(x), [vocab2i[t] for t in s]))
    K = max(len(row) for _, nodes, _ in data for row in nodes)

    B = 32
    data.sort(key=lambda d: d[0])               # similar lengths in the same batch
    batches = []
    for i in range(0, len(data), B):
        chunk = data[i:i + B]
        L = round_up(max(n for n, _, _ in chunk))
        arrays = [to_arrays(nodes, L, K) for _, nodes, _ in chunk]
        gold = np.zeros((len(chunk), L), np.int32)
        for j, (_, _, g) in enumerate(chunk):
            gold[j, :len(g)] = g
        batches.append((
            jnp.array(np.stack([a[0] for a in arrays])),
            jnp.array(np.stack([a[1] for a in arrays])),
            jnp.array(np.stack([a[2] for a in arrays])),
            jnp.array([n for n, _, _ in chunk]),
            jnp.array(gold),
            jnp.array([len(g) for _, _, g in chunk]),
        ))

    # ---- training: same SGD update as before, one step per batch ----
    lr = 1.0
    for epoch in range(10):
        random.shuffle(batches)
        total = 0.0
        for batch in tqdm(batches, desc=f"epoch {epoch}", leave=False):
            loss, grads = batch_loss_and_grad(params, batch)
            params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
            total += float(loss)
        print(f"epoch {epoch}  loss {total / len(batches):.4f}")

    print(viterbi(params, "私は東京都に住む大学院生だ。"))
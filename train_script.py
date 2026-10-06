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
if os.path.exists(kwdlc_path):
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

class TrainCRF:
    def __init__(self, kwdlc):
        # each token: (surface, reading, lemma, pos, pos_subcat, conj_type, conj_form)
        self.sentences = [[tuple(token.values()) for token in sentence] for sentence in kwdlc]

        self.vocab = set()
        for sentence in tqdm(self.sentences, desc="creating vocab"):
            self.vocab.update(sentence)
        self.vocab = sorted(self.vocab)
        self.vocab2i = {token: i for i, token in enumerate(self.vocab)}

        self.surface2ids = defaultdict(list)                 # surface -> token ids with that surface
        for i, token in enumerate(self.vocab):
            self.surface2ids[token[0]].append(i)
        self.max_len = max(len(s) for s in self.surface2ids)

        self.lemma2i      = {v: i for i, v in enumerate(sorted({t[2] for t in self.vocab}))}
        self.pos2i        = {v: i for i, v in enumerate(sorted({t[3] for t in self.vocab}))}
        self.pos_subcat2i = {v: i for i, v in enumerate(sorted({t[4] for t in self.vocab}))}
        self.conj_type2i  = {v: i for i, v in enumerate(sorted({t[5] for t in self.vocab}))}
        self.conj_form2i  = {v: i for i, v in enumerate(sorted({t[6] for t in self.vocab}))}
        self.BOS = self.EOS = len(self.pos2i)
        self.NEG = -1e9                                      # "impossible" score; -inf would give NaN gradients

        # token id -> its 6 feature ids, looked up on the device instead of in Python dicts
        self.FEATURES = ["token", "lemma", "pos", "pos_subcat", "conj_type", "conj_form"]
        self.TOKEN_FEATS = jnp.array([[i, self.lemma2i[t[2]], self.pos2i[t[3]], self.pos_subcat2i[t[4]],
                                self.conj_type2i[t[5]], self.conj_form2i[t[6]]] for i, t in enumerate(self.vocab)])
        self.POS = self.FEATURES.index("pos")

        self.params = {
            "transition": jnp.zeros((len(self.pos2i) + 1, len(self.pos2i) + 1)),
            "emission": {
                "token":      jnp.zeros(len(self.vocab)),
                "lemma":      jnp.zeros(len(self.lemma2i)),
                "pos":        jnp.zeros(len(self.pos2i)),
                "pos_subcat": jnp.zeros(len(self.pos_subcat2i)),
                "conj_type":  jnp.zeros(len(self.conj_type2i)),
                "conj_form":  jnp.zeros(len(self.conj_form2i)),
            },
        }

    def train_crf(self, train=False):
        # Lattice as padded arrays. Row e holds up to K nodes that END at character e:
        #   tok[e, k]   token id        start[e, k]  where the node starts
        #   valid[e, k] real node or padding
        # Row 0 holds BOS at k = 0. The predecessors of node (e, k) are exactly row start[e, k].
        def build_nodes(x):
            nodes = [[] for _ in range(len(x) + 1)]
            nodes[0] = [(0, 0)]                         # BOS placeholder
            for end in range(1, len(x) + 1):
                for start in range(max(0, end - self.max_len), end):
                    if not nodes[start]:
                        continue                        # unreachable from BOS
                    for t in self.surface2ids.get(x[start:end], []):
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
            return sum(params["emission"][f][feats[..., i]] for i, f in enumerate(self.FEATURES))

        def log_Z(params, tok, start, valid, length):
            tok, start, valid = jnp.asarray(tok), jnp.asarray(start), jnp.asarray(valid)
            T = params["transition"]
            feats = self.TOKEN_FEATS[tok]                                    # (L+1, K, 6)
            emit = emission(params, feats)                              # (L+1, K)
            pos = feats[..., self.POS].at[0].set(self.BOS)                        # (L+1, K)
            alpha0 = jnp.full(tok.shape, self.NEG).at[0, 0].set(0.0)         # log alpha(BOS) = 0

            def step(alpha, e):
                prev_alpha = alpha[start[e]]                            # (K, K): row k = predecessors of node k
                prev_pos = pos[start[e]]                                # (K, K)
                cand = prev_alpha + T[prev_pos, pos[e][:, None]]
                new = nn.logsumexp(cand, axis=1) + emit[e]
                return alpha.at[e].set(jnp.where(valid[e], new, self.NEG)), None

            alpha, _ = lax.scan(step, alpha0, jnp.arange(1, tok.shape[0]))
            return nn.logsumexp(alpha[length] + T[pos[length], self.EOS])

        def gold_score(params, gold, n_gold):
            T = params["transition"]
            feats = self.TOKEN_FEATS[gold]                                   # (M, 6)
            mask = jnp.arange(gold.shape[0]) < n_gold
            pos = feats[:, self.POS]
            prev = jnp.concatenate([jnp.array([self.BOS]), pos[:-1]])
            return ((emission(params, feats) + T[prev, pos]) * mask).sum() + T[pos[n_gold - 1], self.EOS]

        def nll(params, tok, start, valid, length, gold, n_gold):
            return log_Z(params, tok, start, valid, length) - gold_score(params, gold, n_gold)

        def l2(params):
            return sum(jnp.sum(w ** 2) for w in jax.tree.leaves(params))

        @jax.jit
        def batch_loss_and_grad(params, batch, lam):
            def objective(p):
                per_sentence = jax.vmap(nll, in_axes=(None, 0, 0, 0, 0, 0, 0))(p, *batch)
                return per_sentence.mean() + 0.5 * lam * l2(p)
            return jax.value_and_grad(objective)(params)

        @jax.jit
        def viterbi_arrays(params, tok, start, valid, length):
            T = params["transition"]
            feats = self.TOKEN_FEATS[tok]
            emit = emission(params, feats)
            pos = feats[..., self.POS].at[0].set(self.BOS)
            delta0 = jnp.full(tok.shape, self.NEG).at[0, 0].set(0.0)

            def step(delta, e):
                cand = delta[start[e]] + T[pos[start[e]], pos[e][:, None]]
                new = cand.max(axis=1) + emit[e]
                return delta.at[e].set(jnp.where(valid[e], new, self.NEG)), cand.argmax(axis=1)

            delta, back = lax.scan(step, delta0, jnp.arange(1, tok.shape[0]))
            return jnp.argmax(delta[length] + T[pos[length], self.EOS]), back

        def viterbi(params, x):
            nodes = build_nodes(x)
            if not nodes[-1]:
                return None                             # some part of x is not in the vocabulary
            L, K = round_up(len(x)), max(len(r) for r in nodes)
            tok, start, valid = to_arrays(nodes, L, K)
            k, back = viterbi_arrays(params, tok, start, valid, len(x))
            k, back, path, e = int(k), np.asarray(back), [], len(x)
            while e > 0:
                path.append(self.vocab[tok[e, k]])
                e, k = start[e, k], back[e - 1, k]
            return path[::-1]

        def round_up(n, m=16):                          # fewer distinct shapes -> fewer recompilations
            return -(-n // m) * m

        import pickle
        # ---- preprocess once: lattices and gold paths as padded arrays ----
        data = []
        for s in tqdm(self.sentences, desc="building lattices"):
            x = "".join(t[0] for t in s)
            data.append((len(x), build_nodes(x), [self.vocab2i[t] for t in s]))
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
        sigma2 = 1.0                                 # prior variance for L2; tune on a dev set
        lam = 1.0 / (sigma2 * len(self.sentences))

        for epoch in range(10):
            random.shuffle(batches)
            total = 0.0
            for batch in tqdm(batches, desc=f"epoch {epoch}", leave=False):
                loss, grads = batch_loss_and_grad(params, batch, lam)
                if train:
                    params = jax.tree.map(lambda w, g: w - lr * g, params, grads)
                total += float(loss)
            print(f"epoch {epoch}  loss {total / len(batches):.4f}")

        crf_index = {
            "vocab":{str(i):list(vocab) for vocab,i in self.vocab2i.items()},
            "surface":self.surface2ids,
            "max_len":self.max_len,
            "lemma":self.lemma2i,
            "pos":self.pos2i,
            "pos_subcat":self.pos_subcat2i,
            "conj_type":self.conj_type2i,
            "conj_form":self.conj_form2i
        }

        with open("crf_index.json", "w", encoding="utf-8") as f:
            json.dump(crf_index, f)

        with open("crf_model.pkl", "wb") as f:
            pickle.dump(params, f)
        
        print(viterbi(params, "私は東京都に住む大学院生だ。"))
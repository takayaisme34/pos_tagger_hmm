# %%
from collections import defaultdict
from parse_knp import iterate_knp_sentences
import jax
import jax.numpy as jnp
from jax import nn

sentences = list(iterate_knp_sentences())

# %%
# --- dictionary: every (word, pos, reading) seen in training is one lattice node type ---
tokens = sorted({(t["word"], t["pos"], t["reading"]) for s in sentences for t in s})
pos_tags = sorted({pos for _, pos, _ in tokens})

tok2i = {tok: i for i, tok in enumerate(tokens)}
pos2i = {pos: i for i, pos in enumerate(pos_tags)}
BOS = EOS = len(pos_tags)                              # one extra index for sentence boundary

tok_pos = jnp.array([pos2i[pos] for _, pos, _ in tokens])   # token id -> pos id
by_surface = defaultdict(list)                         # word -> token ids (one per pos/reading)
for i, (word, _, _) in enumerate(tokens):
    by_surface[word].append(i)
max_len = max(len(w) for w in by_surface)

# %%
# --- parameters: same two tables as the HMM, but free weights instead of probabilities ---
params = {
    "transition": jnp.zeros((len(pos_tags) + 1, len(pos_tags) + 1)),  # [pos1 or BOS, pos2 or EOS]
    "emission":   jnp.zeros(len(tokens)),                              # [(word, pos, reading)]
}

# %%
def build_lattice(text):
    """lattice[end] = [(start, token_id), ...]; only nodes reachable from BOS are kept."""
    lattice = [[] for _ in range(len(text) + 1)]
    for end in range(1, len(text) + 1):
        for start in range(max(0, end - max_len), end):
            if start > 0 and not lattice[start]:
                continue                               # nothing ends here, so no path can reach it
            for t in by_surface.get(text[start:end], []):
                lattice[end].append((start, t))
    return lattice

def gold_score(params, gold):
    """Score of the correct path: sum of its transition and emission weights."""
    pos = [BOS] + [int(tok_pos[t]) for t in gold] + [EOS]
    return params["transition"][jnp.array(pos[:-1]), jnp.array(pos[1:])].sum() \
         + params["emission"][jnp.array(gold)].sum()

def log_Z(params, lattice):
    """Forward algorithm: one log-alpha per node, in order of end position."""
    T, E = params["transition"], params["emission"]
    alpha = [jnp.zeros(1)]                             # log alpha(BOS) = 0
    pos = [jnp.array([BOS])]
    for nodes in lattice[1:]:
        alpha.append(jnp.array([nn.logsumexp(alpha[s] + T[pos[s], tok_pos[t]]) + E[t]
                                for s, t in nodes]))
        pos.append(jnp.array([int(tok_pos[t]) for _, t in nodes], dtype=int))
    return nn.logsumexp(alpha[-1] + T[pos[-1], EOS])

def nll(params, lattice, gold):
    return log_Z(params, lattice) - gold_score(params, gold)

def viterbi(params, text):
    """Same recursion with max instead of logsumexp, plus backpointers."""
    T, E = params["transition"], params["emission"]
    lattice = build_lattice(text)
    best = [[(0.0, None)]]                             # best[end][k] = (score, (prev_end, prev_k))
    pos = [[BOS]]
    for end, nodes in enumerate(lattice[1:], start=1):
        best.append([])
        for s, t in nodes:
            cands = [best[s][k][0] + T[p, tok_pos[t]] for k, p in enumerate(pos[s])]
            k = int(jnp.argmax(jnp.array(cands)))
            best[end].append((cands[k] + E[t], (s, k)))
        pos.append([int(tok_pos[t]) for _, t in nodes])
    final = [sc + T[p, EOS] for (sc, _), p in zip(best[-1], pos[-1])]
    end, k = len(text), int(jnp.argmax(jnp.array(final)))
    path = []
    while end > 0:
        path.append(tokens[lattice[end][k][1]])
        end, k = best[end][k][1]
    return path[::-1]

# %%
# --- training: plain SGD on the negative log-likelihood ---
train = []
for s in sentences:
    text = "".join(t["word"] for t in s)
    gold = [tok2i[(t["word"], t["pos"], t["reading"])] for t in s]
    train.append((build_lattice(text), gold))

grad_fn = jax.value_and_grad(nll)
lr = 0.1
for epoch in range(20):
    total = 0.0
    for lattice, gold in train:
        loss, g = grad_fn(params, lattice, gold)
        params = jax.tree.map(lambda p, gp: p - lr * gp, params, g)
        total += loss
    print(f"epoch {epoch:2d}  loss {total:.3f}")

# %%
print(viterbi(params, "カレーは辛い"))
import itertools
import jax
import jax.numpy as jnp

data = [
    (["the", "cat", "sat"],               ["DET", "NOUN", "VERB"]),
    (["the", "dog", "runs", "quickly"],   ["DET", "NOUN", "VERB", "ADV"]),
    (["a", "quick", "brown", "fox"],      ["DET", "ADJ",  "ADJ",  "NOUN"]),
    (["the", "fox", "jumps"],             ["DET", "NOUN", "VERB"]),
    (["cats", "run", "quickly"],          ["NOUN", "VERB", "ADV"]),
    (["the", "very", "quick", "cat"],     ["DET", "ADV",  "ADJ",  "NOUN"]),
    (["dogs", "sat", "on", "the", "mat"], ["NOUN", "VERB", "PREP", "DET", "NOUN"]),
    (["a", "brown", "dog", "runs"],       ["DET", "ADJ",  "NOUN", "VERB"]),
    (["the", "runner", "jumped"],         ["DET", "NOUN",  "VERB"]),
    (["jumping", "quickly", "is", "hard"],["VERB", "ADV", "VERB", "ADJ"]),
    (["the", "teacher", "ran"],           ["DET", "NOUN", "VERB"]),
    (["very", "quickly", "the", "cat", "jumps"], ["ADV", "ADV", "DET", "NOUN", "VERB"]),
]

# --- index maps (sorted so indices are reproducible) ---
tags  = sorted({t for _, ts in data for t in ts})          # real tags only
vocab = sorted({w for ws, _ in data for w in ws})
tag2i  = {t: i for i, t in enumerate(tags)}
word2i = {w: i for i, w in enumerate(vocab)}
T, V = len(tags), len(vocab)
BOS = T                                                     # extra row for "bos"

# --- params: plain arrays in a dict (a pytree jax.grad understands) ---
params = {
    "transition": jnp.zeros((T + 1, T)),   # [prev_tag (incl. bos), next_tag]
    "emission":   jnp.zeros((V, T)),       # [word, tag]
}

def score(params, words, tags_):
    """Unnormalised log-score of one tag sequence: just add up weights."""
    prev = jnp.concatenate([jnp.array([BOS]), tags_[:-1]])
    return (params["transition"][prev, tags_].sum()
            + params["emission"][words, tags_].sum())

def log_Z_bruteforce(params, words):
    """log sum over ALL tag paths (T^n of them). Fine for this toy data only."""
    n = len(words)
    all_paths = jnp.array(list(itertools.product(range(T), repeat=n)))
    scores = jax.vmap(lambda y: score(params, words, y))(all_paths)
    return jax.nn.logsumexp(scores)

def log_Z(params, words):
    """Same quantity via the forward algorithm: O(n * T^2)."""
    em = params["emission"][words]                     # (n, T)
    alpha = params["transition"][BOS] + em[0]          # (T,)
    for t in range(1, len(words)):
        alpha = jax.nn.logsumexp(alpha[:, None] + params["transition"][:T], axis=0) + em[t]
    return jax.nn.logsumexp(alpha)

def nll(params, words, tags_):
    return log_Z(params, words) - score(params, words, tags_)

def encode(ws, ts):
    return jnp.array([word2i[w] for w in ws]), jnp.array([tag2i[t] for t in ts])

def viterbi(params, words):
    em = params["emission"][words]
    delta = params["transition"][BOS] + em[0]
    back = []
    for t in range(1, len(words)):
        s = delta[:, None] + params["transition"][:T]
        back.append(s.argmax(0))
        delta = s.max(0) + em[t]
    best = [int(delta.argmax())]
    for b in reversed(back):
        best.append(int(b[best[-1]]))
    return [tags[i] for i in reversed(best)]

if __name__ == "__main__":
    encoded = [encode(ws, ts) for ws, ts in data]

    # sanity check: forward algorithm == brute force enumeration
    w, _ = encoded[0]
    print("log Z brute:", log_Z_bruteforce(params, w), " forward:", log_Z(params, w))

    grad_fn = jax.value_and_grad(nll)
    lr = 0.1
    for epoch in range(50):
        total = 0.0
        for w, t in encoded:
            loss, g = grad_fn(params, w, t)
            params = jax.tree.map(lambda p, gp: p - lr * gp, params, g)
            total += loss
        if epoch % 10 == 0:
            print(f"epoch {epoch:2d}  loss {total:.3f}")

    for ws, ts in data[:3]:
        print(ws, "->", viterbi(params, encode(ws, ts)[0]), "gold:", ts)
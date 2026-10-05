
import jax.numpy as jnp
from jax import nn

vocab = ["a", "b", "c", "d", "ab", "cd"]

# scores = {"a":1, "b":1, "c":3, "d":2, "ab":2, "cd":1}
gold_scores = jnp.array([1,1,3,2,2,1])
scores = jnp.zeros(len(vocab))
vocab2i = {word:i for i, word in enumerate(vocab)}

def encode(x):
    return jnp.array([vocab2i[word] for word in x])

def build_lattice(x):
    lattice = [[] for _ in range(len(x)+1)]
    for end in range(len(x)+1):
        for word in vocab:
            str_extract = x[end-len(word):end]
            if str_extract != word:
                continue
            lattice[end].append({"surface":word, "start": end-len(word), "end": end, "prev_words":[]})
    return lattice

def calculate_log_z(lattice, scores):
    for end_position in lattice:                      # in order of end position
        for node in end_position:
            s = scores[vocab2i[node["surface"]]]
            if node["start"] == 0:
                node["alpha"] = s                 # predecessor is BOS, log α(BOS) = 0
            else:
                prev = jnp.array([n["alpha"] for n in lattice[node["start"]]])
                node["alpha"] = nn.logsumexp(prev) + s
    return nn.logsumexp(jnp.array([n["alpha"] for n in lattice[-1]]))



if __name__ == "__main__":
    
    x = "abcd"
    x_encoded = encode(x)
    lattice = build_lattice(x)
    print(lattice)
    #calculate_z(lattice)
    #print(calculate_log_z(lattice, gold_scores))





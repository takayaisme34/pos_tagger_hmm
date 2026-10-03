

data = [
    (["the", "cat", "sat"],               ["DET", "NOUN", "VERB"]),
    (["the", "dog", "runs", "quickly"],   ["DET", "NOUN", "VERB", "ADV"]),
    (["a", "quick", "brown", "fox"],      ["DET", "ADJ",  "ADJ",  "NOUN"]),
    (["the", "fox", "jumps"],             ["DET", "NOUN", "VERB"]),
    (["cats", "run", "quickly"],          ["NOUN", "VERB", "ADV"]),
    (["the", "very", "quick", "cat"],     ["DET", "ADV",  "ADJ",  "NOUN"]),
    (["dogs", "sat", "on", "the", "mat"],["NOUN", "VERB", "PREP", "DET", "NOUN"]),
    (["a", "brown", "dog", "runs"],       ["DET", "ADJ",  "NOUN", "VERB"]),
    (["the", "runner", "jumped"],         ["DET", "NOUN",  "VERB"]),
    (["jumping", "quickly", "is", "hard"],["VERB", "ADV", "VERB", "ADJ"]),
    (["the", "teacher", "ran"],           ["DET", "NOUN", "VERB"]),
    (["very", "quickly", "the", "cat", "jumps"], ["ADV", "ADV", "DET", "NOUN", "VERB"]),
]





import jax
from jax import nn
import jax.numpy as jnp
import math

initializer = nn.initializers.he_normal()
key = jax.random.PRNGKey(0)


pos_tags = set()
pos_tags.add("bos")
vocab = set()
for d in data:
    for pos in d[1]:
        pos_tags.add(pos)
    for word in d[0]:
        vocab.add(word)
pos_tags = sorted(pos_tags)
print(pos_tags)


features = pos_tags

key, k1, k2 = jax.random.split(key, num=3)
transition_weights = initializer(k1, shape=(len(pos_tags), len(pos_tags)))
emission_weights = initializer(k2, shape=(len(vocab), len(features)))

params = {
    "transition" : {
        (pos1, pos2): transition_weights[i,j]
        for i, pos1 in enumerate(pos_tags) for j, pos2 in enumerate(pos_tags)
    },
    "emission": {
        (word, feature): emission_weights[i,j]
        for i, word in enumerate(vocab) for j, feature in enumerate(features)
    }
}


def score(params, sequence):
    sequence = {'surface':'bos','pos':'bos'}
    total_score = 0
    for i in range(0, len(sequence)-1):
        window = sequence[i:i+2]
        t1, t2 = window[0], window[1]
        surface1, pos1 = t1['surface'], t1['pos']
        surface2, pos2 = t2['surface'], t2['pos']
        bigram = params['transition'][pos1][pos2]
        total_score += jnp.log(bigram)

        for feature in features:
            if feature not in t1.values():
                continue
            unigram = params['emission'][surface1][feature]
            total_score += jnp.log(unigram)

    for feature in features:
        if feature not in t2.values():
            continue
        total_score += params['emission'][surface2][feature]

    return total_score



        


if __name__ == "__main__":
    for d in data:
        word_list = d[0]
        tag_list = d[1]
        sequence = [{"surface": word, "pos":tag} for word, tag in zip(word_list, tag_list)]

        numerator = score(sequence, params)

        all_paths = []
        def iterate_paths(word_list):
            for i,word in enumerate(word_list):
                for tag in pos_tags:

                    pass


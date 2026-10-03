

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

initializer = nn.initializers.he_normal()
key = jax.random.PRNGKey(0)


pos_tags = set()
vocab = set()
for d in data:
    for pos in d[1]:
        pos_tags.add(pos)
    for word in d[0]:
        vocab.add(word)
pos_tags = sorted(pos_tags)
print(pos_tags)




    


"""def transition_weights(key, pos_tags):
    weights = initializer(key, shape=(len(pos_tags)**2,1))
    for i in range(len(pos_tags)**2):
        yield weights[i]
transition_weight = transition_weights(k1, pos_tags)"""

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
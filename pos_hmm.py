# %%
from collections import defaultdict
from parse_knp import iterate_knp_sentences
import numpy as np

# POSt-1 -> POSt
transition = defaultdict(lambda: defaultdict(int))
# POS -> word -> reading
emission = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

tokens = set()
words = set()
pos_tags = defaultdict(int)

sentences = list(iterate_knp_sentences())

# %%

for sentence in sentences:
    # per-token stats: run once per token regardless of sentence length
    for token in sentence:
        word, pos, reading = token["word"], token["pos"], token["reading"]
        emission[pos][word][reading] += 1
        pos_tags[pos] += 1
        tokens.add((pos, word, reading))
        words.add(word)

    # per-pair stats: transitions between consecutive tokens only
    for i in range(len(sentence) - 1):
        pos1 = sentence[i]["pos"]
        pos2 = sentence[i + 1]["pos"]
        transition[pos1][pos2] += 1

# ordered by frequency. keep order fixed
pos_tags = [pos for _, pos in sorted(((c, p) for p, c in pos_tags.items()), reverse=True)]

# transition_score[(pos1, pos2)] -> P(pos2 | pos1)
transition_score = {}
for pos1, row in transition.items():
    total = sum(row.values())
    for pos2, count in row.items():
        transition_score[(pos1, pos2)] = count / total


# emission_score[(pos, word, reading)] -> P(word, reading | pos)
emission_score = {}

for pos, word_counts in emission.items():
    total = sum(
        count
        for reading_counts in word_counts.values()
        for count in reading_counts.values()
    )
    for word, reading_counts in word_counts.items():
        for reading, count in reading_counts.items():
            emission_score[(pos, word, reading)] = count / total


def get_transition_score(pos1, pos2, default=0.0):
    return transition_score.get((pos1, pos2), default)

# with safe default for combinations never seen in training
def get_emission_score(pos, word, reading, default=0.0):
    return emission_score.get((pos, word, reading), default)

if __name__ == "__main__":

    print(get_transition_score("名詞", "動詞"))
    print(get_transition_score("名詞", "存在しないタグ"))  # 0.0, unseen
    max_readings = 10
    print(emission["形容詞"]["辛い"])

    print(get_emission_score("形容詞", "辛い", "からい"))
    print(get_emission_score("名詞", "存在しない単語", "なし"))  # 0.0, unseen



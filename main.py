from collections import defaultdict
from parse_knp import iterate_knp_sentences
from pos_hmm import transition_score, emission_score, tokens, pos_tags, get_transition_score, get_emission_score
import math



def build_dictionary(tokens):
    """surface form -> list of (pos, reading)"""
    dic = defaultdict(list)
    for pos, word, reading in tokens:
        dic[word].append((pos, reading))
    max_len = max(len(w) for w in dic)
    return dic, max_len

def create_node(pos, word, reading, start, end):
    return {"pos": pos, "word": word, "reading": reading, "start": start, "end": end}

def build_lattice(input_string, dic, max_len, unk_pos="未知語"):
    n = len(input_string)
    # lattice[i] = nodes ending at i; slot 0 = BOS, slot n+1 = EOS
    lattice = [[] for _ in range(n + 2)]
    BOS_node = create_node("", "BOS", "", 0, 0)
    BOS_node["best_score"] = 0
    lattice[0].append(BOS_node)
    

    for start in range(n):
        found_1char = False
        for length in range(1, min(max_len, n - start) + 1):
            end = start + length
            surface = input_string[start:end]
            for pos, reading in dic.get(surface, []):
                lattice[end].append(create_node(pos, surface, reading, start, end))
                if length == 1:
                    found_1char = True
        # unknown-word fallback: guarantees the lattice is connected
        if not found_1char:
            ch = input_string[start]
            lattice[start + 1].append(create_node(unk_pos, ch, "", start, start + 1))

    lattice[n + 1].append(create_node("", "EOS", "", n, n + 1))
    return lattice


def hmm_viterbi(lattice):

    epsilon = 0.00001
    for end_idx, nodes in enumerate(lattice):
        if end_idx == 0:
            continue
        for node in nodes:
       
            start = node["start"]
            left_neighbors = lattice[start]

            best_u_node = max(left_neighbors, key=lambda node: node["best_score"])
            prev_pos = best_u_node["pos"]
            if prev_pos == "":
                best_u = 0
            else:
                best_u = best_u_node["best_score"]

            trans_score = get_transition_score(prev_pos, node["pos"])
            emis_score = get_emission_score(node["pos"], node["word"], node["reading"], default=1) #log(1) is zero (for unknown words)
            node["best_score"] = best_u + math.log(trans_score + epsilon) + math.log(emis_score + epsilon)
    #print(lattice)

    tagged_sequence = []
    
    current_node = lattice[-2]
    reached_first_index = False
    while not reached_first_index:
        best_node = max(current_node, key=lambda node: node["best_score"])
        tagged_sequence.append(best_node)
        start = best_node["start"]
        current_node = lattice[start]
        if start == 0:
            reached_first_index = True

    #for node in reversed(tagged_sequence):
    #    print(node["word"], node["reading"])

    return list(reversed(tagged_sequence))


    #print("tagged sequence:",reversed(tagged_sequence))


if __name__ == "__main__":

    tokens = set()
    for sentence in iterate_knp_sentences():
        for t in sentence:
            tokens.add((t["pos"], t["word"], t["reading"]))

    dic, max_len = build_dictionary(tokens)
    lattice = build_lattice("あの店は辛いカレーで有名だ", dic, max_len)


    #for i, nodes in enumerate(lattice):
    #    print(i, [(nd["word"], nd["pos"], nd["start"]) for nd in nodes])
    tokenized = hmm_viterbi(lattice)

    print(tokenized)
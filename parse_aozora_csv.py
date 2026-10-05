import csv
import os
from collections import defaultdict
import json

parent_dir = os.path.dirname(os.path.abspath(__file__))
test_csv_path = os.path.join(parent_dir, "KWDLC-1.0", "1567_14913.csv")


group_by_sentence = defaultdict(lambda: defaultdict(list))


header = ["row", "num", "surface", "pos", "pos_subcat", "conj_type", "conj_form", "lemma", "reading", "pronunciation"]
with open(test_csv_path, "r") as f:#, encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        filename = row[0]
        sentence = row[1]
        surface = row[3]
        pos = row[4]
        pos_subcat = row[5]
        lemma = row[10]
        conj_form = row[9]
        conj_type = row[8]
        reading = row[11]

        token = {
            'surface':   surface,
            'reading':   reading,
            'lemma':     lemma,
            'pos':       pos,
            'pos_subcat':pos_subcat,
            'conj_type': conj_type,
            'conj_form': conj_form
        }
        # ['surface','reading','lemma','pos','pos_subcat','conj_type','conj_form']
        group_by_sentence[filename][sentence].append(token)

with open("aozora.jsonl", "w") as f:
    for file in group_by_sentence:
        for sentence in group_by_sentence[file]:
            f.write(
                json.dumps(group_by_sentence[file][sentence]) +
                "\n"
            )
            
            
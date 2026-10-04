import os
from tqdm import tqdm

def parse_knp_pos(text):
    sentences = []
    current_sentence = []

    for line in text.splitlines():

        line = line.strip()

        # End of sentence
        if line == "EOS":
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []

        # Ignore sentence headers
        elif line.startswith("#"):
            continue

        # Ignore bunsetsu and tag headers
        elif line.startswith("*") or line.startswith("+"):
            continue

        # Morpheme
        elif line:
            parts = line.split()
            surface = parts[0]
            reading = parts[1]
            lemma = parts[2]
            pos = parts[3]
            pos_subcategory = parts[4]
            conjugation_type = parts[5]
            conjugation_form = parts[6]

            current_sentence.append({
                "word": surface,
                "reading": reading,
                "lemma":lemma,
                "pos": pos,
                "pos_subcategory": pos_subcategory,
                "conjugation_type": conjugation_type,
                "conjugation_form": conjugation_form
            })

    return sentences



def iterate_knp_sentences():

    knp_directory = os.path.join("KWDLC-1.0", "dat", "rel")

    knp_folders = os.listdir(knp_directory)

    with tqdm(
        total=len(knp_folders),
        desc="Iterating KNP folders"
    ) as progress_bar:

        for folder_name in knp_folders:

            folder_path = os.path.join(
                knp_directory,
                folder_name
            )

            progress_bar.update(1)

            for file_name in os.listdir(folder_path):

                file_path = os.path.join(
                    folder_path,
                    file_name
                )

                with open(
                    file_path,
                    "rt",
                    encoding="utf-8"
                ) as file:

                    sentences = parse_knp_pos(file.read())

                for sentence in sentences:
                    yield sentence



if __name__ == "__main__":
    sentences = list(iterate_knp_sentences())
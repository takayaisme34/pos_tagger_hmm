# A tiny fake lattice: lattice[i] = nodes whose span ends at position i
# Each node just needs "start" and a fake "score" for this demo.
lattice = {
    0: [{"word": "BOS", "start": 0, "best": 0}],          # base case
    1: [{"word": "A",   "start": 0, "score": 3}],
    2: [{"word": "AA",  "start": 0, "score": 5},
        {"word": "B",   "start": 1, "score": 1}],
    3: [{"word": "C",   "start": 2, "score": 2}],
}

# Process slots in increasing order of "end" (0, 1, 2, 3, ...)
for end in sorted(lattice.keys()):
    if end == 0:
        continue  # BOS is already the base case

    for node in lattice[end]:
        start = node["start"]          # <-- this number IS the alignment
        left_neighbours = lattice[start]  # look up that earlier slot

        print(start, left_neighbours)

        # every left neighbour must already have "best" set,
        # because start < end, and we did smaller ends first
        best_so_far = max(u["best"] + node["score"] for u in left_neighbours)
        node["best"] = best_so_far

        #print(f"{node['word']} (start={start}, end={end}) "
        #      f"-> pulled from slot {start} -> best={node['best']}")
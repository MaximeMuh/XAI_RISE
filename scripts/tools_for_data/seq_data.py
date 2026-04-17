from pathlib import Path


def read_fasta(path):
    path = Path(path)
    seqs = []
    header = None
    buf = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seqs.append((header, "".join(buf)))
            header = line[1:].strip()
            buf = []
        else:
            buf.append(line)
    if header is not None:
        seqs.append((header, "".join(buf)))
    return seqs

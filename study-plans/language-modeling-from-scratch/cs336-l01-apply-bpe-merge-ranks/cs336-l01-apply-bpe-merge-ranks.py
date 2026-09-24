def apply_merge(seq, left, right, new_id):
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left and seq[i+1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    
    return out

def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns token IDs after applying the ordered merge rules.
    """
    
    seq = list(text.encode("utf-8"))
    for merge in merges:
        left, right, new_id = merge
        seq = apply_merge(seq, left, right, new_id)
    return seq

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns the text reconstructed from the token bytes.
    """
    byte_list = []
    for tid in ids:
        byte_list.extend(vocab[tid])
    return bytes(byte_list).decode('utf-8')
